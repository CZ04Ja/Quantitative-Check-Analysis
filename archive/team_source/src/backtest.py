"""
Lightweight vectorised intraday backtester for the QTC research project.

Design goals:
  - Single-asset long/short
  - Entry signals produced by the caller (boolean arrays or integer -1/0/+1)
  - Executes at next bar's OPEN (no look-ahead)
  - Optional execution guardrails:
      * cooldown between trades (minutes)
      * no new entry in the last `flat_before_close` minutes
      * forced flatten at end of each NY session
      * no pyramiding (only one open position at a time)
  - Optional exit rules:
      * fixed or volatility-scaled stop-loss
      * max holding time in bars
  - Configurable per-trade slippage / commission in bps

Outputs a per-bar DataFrame with columns:
    position, trade_id, entry_price, exit_price, pnl_bp, returns
And a trade log DataFrame with per-trade PnL and holding time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    # Execution guardrails
    cooldown_bars: int = 5           # min bars between consecutive entries
    flat_before_close_min: int = 15  # disallow new entries in last N minutes of session
    forced_flatten: bool = True      # force-close at session end
    no_pyramiding: bool = True       # at most one position at a time

    # Exit rules
    stop_loss_k: Optional[float] = None   # stop = k * rolling_vol * entry_price. None = no stop.
    max_hold_bars: Optional[int] = None   # time-based exit
    trailing: bool = False                # use trailing stop instead of fixed

    # Cost model
    slippage_bps: float = 1.0        # per-side slippage in bps
    commission_bps: float = 0.0      # per-side commission in bps

    # Sizing
    notional: float = 10_000.0       # base dollar notional per trade (B0/B1)
    use_vol_targeting: bool = False  # if True, scale notional by 1/rolling_vol
    vol_target_daily: float = 0.02   # target daily vol (2%)
    kelly_fraction: float = 0.0      # 0 disables Kelly; 0.25 means 25% of full Kelly


def _session_end_mask(ts_ny: pd.Series, minutes_before_close: int) -> pd.Series:
    """True on bars that are within `minutes_before_close` of the 16:00 NY close."""
    mins = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    close_mins = 16 * 60
    return (mins >= close_mins - minutes_before_close) & (mins < close_mins)


def _session_last_bar_mask(ts_ny: pd.Series) -> pd.Series:
    """True on the final bar of each NY trading day (to force-flatten)."""
    date = ts_ny.dt.date
    return date != date.shift(-1)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cfg: BacktestConfig,
    size_override: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    df : per-bar DataFrame. Must contain: timestamp (UTC), open, close,
         rolling_vol. Ideally sorted by timestamp.
    signal : integer series aligned with df.index.
         +1 = want long, -1 = want short, 0 = no signal.
    cfg : BacktestConfig
    size_override : optional per-bar multiplier for position size.
                    If None, uses cfg.notional (optionally vol-scaled).

    Returns
    -------
    bars : DataFrame of per-bar state (position, pnl_bp, returns, equity).
    trades : DataFrame of per-trade records.
    """
    d = df.reset_index(drop=True).copy()
    sig = pd.Series(signal, index=d.index).fillna(0).astype(int).values

    # NY timestamp for session rules
    ts_ny = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
    ny_series = pd.Series(ts_ny.values, index=d.index)
    blackout = _session_end_mask(ny_series, cfg.flat_before_close_min).values
    last_bar = _session_last_bar_mask(ny_series).values

    open_px = d["open"].values
    close_px = d["close"].values
    vol = d.get("rolling_vol", pd.Series(np.nan, index=d.index)).values
    has_vol = np.any(~np.isnan(vol)) and cfg.stop_loss_k is not None

    # Size per bar
    if size_override is not None:
        base_size = pd.Series(size_override, index=d.index).fillna(0).values.astype(float)
    else:
        if cfg.use_vol_targeting and has_vol:
            scale = np.where(vol > 0, cfg.vol_target_daily / (vol * np.sqrt(390)), 1.0)
            scale = np.clip(scale, 0.0, 5.0)
            base_size = cfg.notional * scale
        else:
            base_size = np.full(len(d), cfg.notional, dtype=float)
        if cfg.kelly_fraction > 0 and "kelly_mu" in d.columns and "kelly_sigma" in d.columns:
            mu = d["kelly_mu"].fillna(0).values
            sd = d["kelly_sigma"].replace(0, np.nan).values
            frac = np.where(sd > 0, mu / (sd**2), 0)
            frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
            frac = np.clip(frac * cfg.kelly_fraction, 0.0, 1.0)
            base_size = base_size * frac

    position = np.zeros(len(d), dtype=int)
    entry_px = np.zeros(len(d))
    size = np.zeros(len(d))
    trade_id = np.full(len(d), -1, dtype=int)
    pnl_bp = np.zeros(len(d))

    cur_pos = 0
    cur_entry = 0.0
    cur_size = 0.0
    bars_since_exit = 10**6
    bars_held = 0
    active_trade = -1
    peak_since_entry = 0.0
    trades = []
    trip_cost_bps = cfg.slippage_bps + cfg.commission_bps

    for i in range(len(d) - 1):
        # 1) Bar state before trading decision
        if cur_pos != 0:
            bars_held += 1

        # 2) Stop-loss check (based on THIS bar's close)
        exit_now = False
        exit_reason = None
        if cur_pos != 0 and cfg.stop_loss_k is not None and has_vol:
            v = vol[i]
            if v > 0 and np.isfinite(v):
                stop_width = cfg.stop_loss_k * v
                if cfg.trailing:
                    if cur_pos > 0:
                        peak_since_entry = max(peak_since_entry, close_px[i])
                        stop_level = peak_since_entry * (1 - stop_width)
                        if close_px[i] <= stop_level:
                            exit_now = True
                            exit_reason = "trail_stop"
                    else:
                        peak_since_entry = min(peak_since_entry, close_px[i]) if peak_since_entry else close_px[i]
                        stop_level = peak_since_entry * (1 + stop_width)
                        if close_px[i] >= stop_level:
                            exit_now = True
                            exit_reason = "trail_stop"
                else:
                    if cur_pos > 0 and close_px[i] <= cur_entry * (1 - stop_width):
                        exit_now = True
                        exit_reason = "stop"
                    elif cur_pos < 0 and close_px[i] >= cur_entry * (1 + stop_width):
                        exit_now = True
                        exit_reason = "stop"

        if cur_pos != 0 and cfg.max_hold_bars is not None and bars_held >= cfg.max_hold_bars:
            exit_now = True
            exit_reason = exit_reason or "max_hold"

        # 3) Forced flatten at end of session
        force_flat = cfg.forced_flatten and last_bar[i]
        if cur_pos != 0 and force_flat:
            exit_now = True
            exit_reason = exit_reason or "eod_flat"

        # 4) Signal-driven exits: flip / zero
        desired = sig[i]
        if cur_pos != 0 and not exit_now:
            if desired == 0 or (desired != 0 and np.sign(desired) != np.sign(cur_pos)):
                exit_now = True
                exit_reason = exit_reason or ("flip" if desired != 0 else "flat_signal")

        # 5) Execute exit at the NEXT bar's open
        if exit_now:
            fill = open_px[i + 1]
            ret = (fill - cur_entry) / cur_entry * np.sign(cur_pos)
            ret -= 2 * trip_cost_bps / 10_000  # round-trip cost
            pnl_bp[i + 1] = ret * 10_000
            trades.append({
                "trade_id": active_trade,
                "entry_bar": int(np.where(trade_id == active_trade)[0][0]) if np.any(trade_id == active_trade) else i,
                "exit_bar": i + 1,
                "entry_time": d["timestamp"].iloc[trade_id.tolist().index(active_trade)] if active_trade in trade_id.tolist() else d["timestamp"].iloc[i],
                "exit_time": d["timestamp"].iloc[i + 1],
                "direction": int(cur_pos),
                "entry_px": cur_entry,
                "exit_px": fill,
                "size": cur_size,
                "ret": ret,
                "pnl_usd": ret * cur_size,
                "hold_bars": bars_held,
                "exit_reason": exit_reason,
            })
            cur_pos = 0
            cur_entry = 0.0
            cur_size = 0.0
            bars_since_exit = 0
            bars_held = 0
            peak_since_entry = 0.0
            active_trade = -1

        # 6) Consider entering
        if (
            cur_pos == 0
            and desired != 0
            and bars_since_exit >= cfg.cooldown_bars
            and not blackout[i]
        ):
            if base_size[i] > 0:
                cur_pos = int(np.sign(desired))
                cur_entry = open_px[i + 1]
                cur_size = base_size[i]
                bars_held = 0
                peak_since_entry = cur_entry
                active_trade = i + 1
                trade_id[i + 1] = active_trade

        position[i] = cur_pos
        entry_px[i] = cur_entry
        size[i] = cur_size
        if active_trade >= 0 and trade_id[i] == -1:
            trade_id[i] = active_trade
        bars_since_exit += 1

    # Final bar: force flatten any open
    i = len(d) - 1
    if cur_pos != 0:
        fill = close_px[i]
        ret = (fill - cur_entry) / cur_entry * np.sign(cur_pos)
        ret -= 2 * trip_cost_bps / 10_000
        pnl_bp[i] = ret * 10_000
        trades.append({
            "trade_id": active_trade,
            "entry_bar": trade_id.tolist().index(active_trade) if active_trade in trade_id.tolist() else i,
            "exit_bar": i,
            "entry_time": d["timestamp"].iloc[trade_id.tolist().index(active_trade)] if active_trade in trade_id.tolist() else d["timestamp"].iloc[i],
            "exit_time": d["timestamp"].iloc[i],
            "direction": int(cur_pos),
            "entry_px": cur_entry,
            "exit_px": fill,
            "size": cur_size,
            "ret": ret,
            "pnl_usd": ret * cur_size,
            "hold_bars": bars_held,
            "exit_reason": "final_flat",
        })

    bars = pd.DataFrame({
        "timestamp": d["timestamp"].values,
        "close": close_px,
        "position": position,
        "trade_id": trade_id,
        "pnl_bp": pnl_bp,
    })
    bars["ret"] = bars["pnl_bp"] / 10_000
    bars["equity_ret"] = bars["ret"].cumsum()

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("exit_bar").reset_index(drop=True)

    return bars, trades_df


def annualization_factor(bars_per_day: float = 390, trading_days: float = 252) -> float:
    return np.sqrt(bars_per_day * trading_days)


def compute_metrics(bars: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """
    Aggregate evaluation metrics used across the research journal.
    Returns a flat dict (one row of the ablation summary table).
    """
    m: dict = {}
    if bars.empty:
        return m

    # Per-bar returns (only populated on exit bars, rest are 0)
    rets = bars["ret"].values
    eq = bars["equity_ret"].values
    n_bars = len(bars)
    n_days = max(1, n_bars / 390)

    # Cumulative / annualised
    total_ret = eq[-1]
    ann = 252 / n_days
    m["total_return"] = float(total_ret)
    m["ann_return"] = float(total_ret * ann)

    # Sharpe / Sortino using per-bar returns (0 on no-action bars)
    std = rets.std(ddof=0)
    m["sharpe"] = float(rets.mean() / std * annualization_factor()) if std > 0 else 0.0
    downside = rets[rets < 0]
    ds = downside.std(ddof=0) if len(downside) else 0.0
    m["sortino"] = float(rets.mean() / ds * annualization_factor()) if ds > 0 else 0.0

    # Max drawdown on cumulative curve
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    m["max_drawdown"] = float(dd.min())
    m["calmar"] = float(m["ann_return"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 else 0.0

    # Trade-level
    if trades.empty:
        m.update({
            "n_trades": 0, "win_rate": np.nan, "avg_hold_bars": np.nan,
            "avg_winner": np.nan, "avg_loser": np.nan, "winner_loser_ratio": np.nan,
            "right_tail_retention": np.nan, "stop_out_frequency": np.nan,
            "turnover_x": 0.0,
        })
        return m

    t = trades
    winners = t[t["ret"] > 0]
    losers = t[t["ret"] <= 0]
    m["n_trades"] = int(len(t))
    m["win_rate"] = float(len(winners) / len(t))
    m["avg_hold_bars"] = float(t["hold_bars"].mean())
    m["avg_winner"] = float(winners["ret"].mean()) if len(winners) else 0.0
    m["avg_loser"] = float(losers["ret"].mean()) if len(losers) else 0.0
    m["winner_loser_ratio"] = float(abs(m["avg_winner"] / m["avg_loser"])) if m["avg_loser"] < 0 else np.nan

    # Right-tail retention: top-5% trades' contribution to total positive PnL
    pos_pnl = winners["ret"].sum()
    if pos_pnl > 0 and len(winners) >= 20:
        cutoff = winners["ret"].quantile(0.95)
        top_tail = winners.loc[winners["ret"] >= cutoff, "ret"].sum()
        m["right_tail_retention"] = float(top_tail / pos_pnl)
    else:
        m["right_tail_retention"] = np.nan

    # Stop-out frequency
    m["stop_out_frequency"] = float((t["exit_reason"].isin(["stop", "trail_stop"])).mean())

    # Turnover proxy = n_trades / n_days
    m["turnover_x"] = float(len(t) / n_days)
    return m


def build_baseline_signal(panel: pd.DataFrame) -> pd.Series:
    """B0 signal: +1 when long_signal_baseline == 1, -1 when short_signal_baseline == 1, else 0."""
    s = pd.Series(0, index=panel.index, dtype=int)
    s.loc[panel["long_signal_baseline"] == 1] = 1
    s.loc[panel["short_signal_baseline"] == 1] = -1
    return s


def build_alpha_confirmed_signal(
    panel: pd.DataFrame,
    vwap_dev_k: float = 0.0005,
    vol_surprise_min: float = 1.0,
    require_proxy_align: bool = True,
) -> pd.Series:
    """
    M1 signal: baseline trend direction AND microstructure confirmation.
      * VWAP deviation same sign as trend (avoid entering when trend disagrees with intraday flow)
      * volume_surprise above a floor (avoid dead tape)
      * proxy (SPY/QQQ/XLK) also in same direction, if requested
    """
    long_base = panel["long_signal_baseline"] == 1
    short_base = panel["short_signal_baseline"] == 1

    vwap = panel.get("vwap_dev", pd.Series(np.nan, index=panel.index))
    vol_surp = panel.get("volume_surprise", pd.Series(1.0, index=panel.index)).fillna(1.0)
    vol_ok = vol_surp >= vol_surprise_min

    long_ok = long_base & (vwap > vwap_dev_k) & vol_ok
    short_ok = short_base & (vwap < -vwap_dev_k) & vol_ok

    if require_proxy_align:
        # At least one proxy agreeing
        proxy_up = pd.Series(False, index=panel.index)
        proxy_dn = pd.Series(False, index=panel.index)
        for px in ["SPY", "QQQ", "XLK"]:
            up_col = f"{px}_is_uptrend"
            dn_col = f"{px}_is_downtrend"
            if up_col in panel.columns:
                proxy_up = proxy_up | (panel[up_col] == 1)
            if dn_col in panel.columns:
                proxy_dn = proxy_dn | (panel[dn_col] == 1)
        long_ok = long_ok & proxy_up
        short_ok = short_ok & proxy_dn

    s = pd.Series(0, index=panel.index, dtype=int)
    s.loc[long_ok] = 1
    s.loc[short_ok] = -1
    return s


def apply_regime_gate(signal: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Simple rule-based regime gate: block entries during high-vol AND not-trending bars."""
    high_vol = panel.get("is_high_vol", 0).astype(bool)
    strong_trend = panel.get("is_strong_trend", 0).astype(bool)
    # Allow trades when: low-vol OR (high-vol AND strong trend)
    allow = (~high_vol) | (high_vol & strong_trend)
    out = signal.copy()
    out.loc[~allow] = 0
    return out


def triple_barrier_labels(
    panel: pd.DataFrame,
    upper_k: float = 2.0,
    lower_k: float = 1.5,
    max_hold_bars: int = 30,
    vol_col: str = "rolling_vol",
    price_col: str = "close",
) -> pd.Series:
    """
    For each bar i, looking forward up to `max_hold_bars`, return:
      +1 if upper barrier (entry * (1 + upper_k * vol)) hit first
      -1 if lower barrier hit first
       0 if horizon expires without hitting either
    Used as meta-label target for the primary baseline signal.
    """
    p = panel[price_col].values
    v = panel[vol_col].values
    n = len(panel)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        vi = v[i]
        if not np.isfinite(vi) or vi <= 0:
            continue
        upper = p[i] * (1 + upper_k * vi)
        lower = p[i] * (1 - lower_k * vi)
        end = min(n, i + max_hold_bars + 1)
        window = p[i + 1:end]
        hit_up = np.where(window >= upper)[0]
        hit_dn = np.where(window <= lower)[0]
        if len(hit_up) == 0 and len(hit_dn) == 0:
            labels[i] = 0
        elif len(hit_up) == 0:
            labels[i] = -1
        elif len(hit_dn) == 0:
            labels[i] = 1
        else:
            labels[i] = 1 if hit_up[0] < hit_dn[0] else -1
    return pd.Series(labels, index=panel.index, dtype=int)


def pretty_metrics_row(name: str, m: dict) -> dict:
    """Format a metrics dict as a single-row record for the summary table."""
    return {
        "version": name,
        "ann_return": f"{m.get('ann_return', 0):.2%}",
        "sharpe": f"{m.get('sharpe', 0):.2f}",
        "sortino": f"{m.get('sortino', 0):.2f}",
        "max_dd": f"{m.get('max_drawdown', 0):.2%}",
        "calmar": f"{m.get('calmar', 0):.2f}",
        "n_trades": m.get("n_trades", 0),
        "win_rate": f"{m.get('win_rate', np.nan):.1%}" if not pd.isna(m.get("win_rate")) else "—",
        "avg_hold": f"{m.get('avg_hold_bars', np.nan):.1f}" if not pd.isna(m.get("avg_hold_bars")) else "—",
        "w/l_ratio": f"{m.get('winner_loser_ratio', np.nan):.2f}" if not pd.isna(m.get("winner_loser_ratio")) else "—",
        "right_tail": f"{m.get('right_tail_retention', np.nan):.1%}" if not pd.isna(m.get("right_tail_retention")) else "—",
        "stop_freq": f"{m.get('stop_out_frequency', np.nan):.1%}" if not pd.isna(m.get("stop_out_frequency")) else "—",
        "turnover/d": f"{m.get('turnover_x', 0):.1f}",
    }
