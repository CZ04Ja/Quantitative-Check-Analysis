"""
Cross-sectional residual reversal strategy for the QTC research project.

Academic foundation:
  - Nagel (2012) "Evaporating Liquidity" - short-term reversal as a liquidity premium
  - Avellaneda & Lee (2010) "Statistical Arbitrage in the U.S. Equities Market"
     - residualise returns against a factor, trade the residual z-score
  - Heston, Korajczyk & Sadka (2010) "Intraday Patterns in the Cross-section of
     Stock Returns" - intraday seasonality motivates time-of-day filters
  - Lopez de Prado (2018) Ch. 3 - triple-barrier meta-labeling

Core idea:
  1) Build a tight universe of tech single names + a sector proxy (XLK).
  2) For each bar, regress each stock's 1-min return on the proxy return
     using a rolling window. Keep the residual - this is the stock-specific
     return after removing common factor noise.
  3) Accumulate residuals over the last H minutes. Compute a cross-sectional
     z-score across the universe.
  4) Bet AGAINST the extremes (negative z => long, positive z => short).
     This is mean reversion of the idiosyncratic component.
  5) Normalise weights so the book is dollar-neutral with a fixed gross.
  6) Apply regime and time-of-day gates, vol targeting, and an optional
     meta-label classifier to filter low-confidence trades.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_wide_panels(
    panel: pd.DataFrame,
    target_symbols: list[str],
    proxy_symbol: str = "XLK",
) -> dict[str, pd.DataFrame]:
    """
    Reshape the long research panel into a set of wide DataFrames indexed by
    timestamp, with one column per target symbol, plus a proxy series.

    Returns dict with keys:
        close, open, high, low, volume, ret_1m, rolling_vol, vwap_dev,
        volume_surprise, trend_strength, is_high_vol, is_strong_trend,
        proxy_close, proxy_ret_1m, proxy_rolling_vol
    """
    all_needed = target_symbols + [proxy_symbol]
    sub = panel[panel["symbol"].isin(all_needed)].copy()

    def _pivot(field: str) -> pd.DataFrame:
        return (
            sub.pivot_table(index="timestamp", columns="symbol", values=field)
            .sort_index()
        )

    out: dict[str, pd.DataFrame] = {}
    for f in ["open", "high", "low", "close", "volume", "ret_1m",
              "rolling_vol", "vwap_dev", "volume_surprise", "trend_strength",
              "is_high_vol", "is_strong_trend", "fwd_ret_15m"]:
        if f in sub.columns:
            out[f] = _pivot(f)

    # Keep only timestamps where ALL targets plus proxy have data
    common_idx = None
    for f in ["close", "ret_1m"]:
        df = out[f]
        has_all = df[all_needed].notna().all(axis=1)
        common_idx = has_all if common_idx is None else (common_idx & has_all)
    common_idx = common_idx[common_idx].index
    for k in list(out.keys()):
        out[k] = out[k].reindex(common_idx)

    # Split proxy out for convenience
    out["proxy_close"] = out["close"][proxy_symbol]
    out["proxy_ret_1m"] = out["ret_1m"][proxy_symbol]
    out["proxy_rolling_vol"] = out["rolling_vol"][proxy_symbol]

    # Restrict target columns
    for k in ["open", "high", "low", "close", "volume", "ret_1m", "rolling_vol",
              "vwap_dev", "volume_surprise", "trend_strength",
              "is_high_vol", "is_strong_trend", "fwd_ret_15m"]:
        if k in out:
            out[k] = out[k][target_symbols]

    return out


# ---------------------------------------------------------------------------
# Factor residualisation
# ---------------------------------------------------------------------------

def rolling_beta(
    ret_wide: pd.DataFrame,
    proxy_ret: pd.Series,
    window: int = 390,
    min_periods: int = 200,
) -> pd.DataFrame:
    """
    Rolling OLS beta of each stock's return series on the proxy, with
    no-look-ahead (shifts to use only info up to t-1).
    Assumes zero-mean returns (valid for 1-min bars).
    """
    num = ret_wide.mul(proxy_ret, axis=0).rolling(window, min_periods=min_periods).sum().shift(1)
    den = proxy_ret.pow(2).rolling(window, min_periods=min_periods).sum().shift(1)
    beta = num.div(den.replace(0, np.nan), axis=0)
    return beta


def residualize(
    ret_wide: pd.DataFrame,
    proxy_ret: pd.Series,
    window: int = 390,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (residual_returns, betas)."""
    beta = rolling_beta(ret_wide, proxy_ret, window=window)
    resid = ret_wide.sub(beta.mul(proxy_ret, axis=0))
    return resid, beta


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def xsec_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    m = wide.mean(axis=1)
    s = wide.std(axis=1, ddof=0).replace(0, np.nan)
    z = wide.sub(m, axis=0).div(s, axis=0)
    return z


def ts_zscore(wide: pd.DataFrame, window: int = 390, min_periods: int = 60) -> pd.DataFrame:
    m = wide.rolling(window, min_periods=min_periods).mean().shift(1)
    s = wide.rolling(window, min_periods=min_periods).std(ddof=0).shift(1).replace(0, np.nan)
    return (wide - m) / s


def build_reversal_weights(
    score: pd.DataFrame,
    mode: str = "continuous",
    target_gross: float = 1.0,
    z_clip: float = 3.0,
    n_long: int = 2,
    n_short: int = 2,
) -> pd.DataFrame:
    """
    Convert a cross-sectional z-score matrix into portfolio weights.

    mode="continuous": weights proportional to -z (clipped), scale so sum(|w|) = target_gross.
    mode="rank":       long bottom n_long (smallest z), short top n_short (largest z), equal weight.
    """
    if mode == "continuous":
        raw = -score.clip(-z_clip, z_clip)
        denom = raw.abs().sum(axis=1).replace(0, np.nan)
        w = raw.div(denom, axis=0).mul(target_gross)
        return w.fillna(0)

    if mode == "rank":
        ranks = score.rank(axis=1, method="first", ascending=True)
        n = score.notna().sum(axis=1)
        w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        for t, row in ranks.iterrows():
            if row.isna().all():
                continue
            n_t = int(n.loc[t])
            nl, ns = min(n_long, n_t // 2), min(n_short, n_t // 2)
            if nl == 0 or ns == 0:
                continue
            longs = row.nsmallest(nl).index
            shorts = row.nlargest(ns).index
            w.loc[t, longs] = target_gross / (2 * nl)
            w.loc[t, shorts] = -target_gross / (2 * ns)
        return w

    raise ValueError(f"unknown mode={mode!r}")


def apply_holding_period(weights: pd.DataFrame, hold_bars: int) -> pd.DataFrame:
    """
    Overlapping portfolio trick: average the last `hold_bars` target weights.
    This is equivalent to running `hold_bars` parallel portfolios offset by
    one bar each, which reduces turnover and noise without changing expected
    return per unit risk.
    """
    return weights.rolling(hold_bars, min_periods=1).mean()


def apply_time_of_day_filter(
    weights: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    skip_first_min: int = 30,
    skip_last_min: int = 15,
) -> pd.DataFrame:
    """Zero-out positions in the first `skip_first_min` and last `skip_last_min`
    minutes of each NY session."""
    ts = pd.to_datetime(timestamps, utc=True).tz_convert("America/New_York")
    mins = ts.hour * 60 + ts.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    keep = (mins >= open_min + skip_first_min) & (mins < close_min - skip_last_min)
    mask = pd.Series(keep, index=timestamps)
    return weights.mul(mask.astype(float), axis=0)


def apply_vol_regime_filter(
    weights: pd.DataFrame,
    proxy_vol: pd.Series,
    min_pct: float = 0.25,
    max_pct: float = 0.95,
    lookback: int = 390 * 20,
) -> pd.DataFrame:
    """
    Only trade when proxy realised vol is between `min_pct` and `max_pct` of
    its trailing distribution. Very low vol -> no noise to mean-revert; very
    high vol -> genuine trend / event risk.
    """
    q_lo = proxy_vol.rolling(lookback, min_periods=lookback // 4).quantile(min_pct).shift(1)
    q_hi = proxy_vol.rolling(lookback, min_periods=lookback // 4).quantile(max_pct).shift(1)
    allow = (proxy_vol >= q_lo) & (proxy_vol <= q_hi)
    return weights.mul(allow.reindex(weights.index).fillna(False).astype(float), axis=0)


def sparse_reversion_state_machine(
    signal_z: pd.DataFrame,
    z_in: float = 2.0,
    z_out_ratio: float = 0.2,
    hold_bars: int = 60,
    gross: float = 1.0,
) -> pd.DataFrame:
    """
    Per-stock state machine that implements sparse contrarian trading:
      * Enter long  when z <= -z_in
      * Enter short when z >= +z_in
      * Exit when |z| < z_in * z_out_ratio, or after `hold_bars`, or when
        the sign flips to the opposite side of zero.
      * No pyramiding: at most one position per stock at a time.
    Weights are normalised each bar so sum(|w|) <= gross.
    This dramatically lowers turnover versus an every-bar rebalance.
    """
    T, N = signal_z.shape
    positions = np.zeros((T, N), dtype=float)
    sz = signal_z.fillna(0).values
    z_exit = z_in * z_out_ratio
    cur = np.zeros(N)
    held = np.zeros(N, dtype=int)
    for t in range(T):
        for s in range(N):
            zt = sz[t, s]
            if cur[s] != 0:
                held[s] += 1
                exit_cond = (
                    abs(zt) < z_exit
                    or held[s] >= hold_bars
                    or (cur[s] > 0 and zt > 0)
                    or (cur[s] < 0 and zt < 0)
                )
                if exit_cond:
                    cur[s] = 0
                    held[s] = 0
            if cur[s] == 0:
                if zt <= -z_in:
                    cur[s] = 1
                    held[s] = 0
                elif zt >= z_in:
                    cur[s] = -1
                    held[s] = 0
            positions[t, s] = cur[s]

    w = pd.DataFrame(positions, index=signal_z.index, columns=signal_z.columns)
    na = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(na, axis=0).mul(gross).fillna(0)


def dispersion_regime_filter(
    resid_wide: pd.DataFrame,
    fast_days: int = 5,
    slow_days: int = 60,
    bars_per_day: int = 390,
) -> pd.Series:
    """
    Regime detector: mean-reversion works when cross-sectional residual
    dispersion is elevated (stocks decorrelate -> there is idiosyncratic
    mispricing to harvest). When dispersion collapses (coordinated market
    move), there is no reversion fuel.

    Returns a boolean series; True = on (trade), False = off (flat).
    All rolling statistics are strictly lagged -> no look-ahead.
    """
    fast = resid_wide.rolling(bars_per_day * fast_days).std(ddof=0).mean(axis=1).shift(1)
    slow = fast.rolling(bars_per_day * slow_days).median().shift(1)
    return fast > slow


def breadth_regime_filter(
    resid_wide: pd.DataFrame,
    fast_days: int = 5,
    slow_days: int = 60,
    bars_per_day: int = 390,
) -> dict[str, pd.Series]:
    """
    Breadth-aware regime detector (Iteration F, Phase 12).

    The vanilla dispersion gate can be fooled by a single-stock shock
    (e.g. earnings on NVDA) pulling the mean residual std up while the
    other three stocks have no reversion setup.

    Compute:
        fast(t)  = mean_stock( rolling-5d std residual )
        fast_median = rolling-60d median of fast
        concentration(t)   = std_stock(rolling std) / mean_stock(rolling std)
        concentration_median = rolling-60d median of concentration

    Returns a dict with three boolean series, strictly lagged:
        disp_on    = fast > fast_median               (baseline)
        broad_on   = concentration < concentration_median
        combined   = disp_on AND broad_on             (iteration F2)
    """
    roll_std = resid_wide.rolling(bars_per_day * fast_days).std(ddof=0)
    fast = roll_std.mean(axis=1).shift(1)
    fast_slow = fast.rolling(bars_per_day * slow_days).median().shift(1)
    conc = (roll_std.std(axis=1) / roll_std.mean(axis=1).replace(0, np.nan)).shift(1)
    conc_slow = conc.rolling(bars_per_day * slow_days).median().shift(1)
    disp_on = (fast > fast_slow)
    broad_on = (conc < conc_slow)
    return {
        "disp_on": disp_on,
        "broad_on": broad_on,
        "combined": (disp_on & broad_on),
    }


def per_stock_vol_mask(
    resid_wide: pd.DataFrame,
    fast_days: int = 5,
    slow_days: int = 60,
    bars_per_day: int = 390,
) -> pd.DataFrame:
    """
    Per-stock idiosyncratic-vol gate (Iteration G, Phase 12).

    Only allow a position in stock s at bar t when s's own recent
    realised residual-std is above its own trailing slow median. This
    prevents holding a stock that has gone quiescent (no reversion
    energy on that specific name) even when the cross-section is
    broadly noisy.

    All rolling statistics are strictly lagged; returns a 0/1 float
    matrix alignable to the weight matrix.
    """
    roll_std = resid_wide.rolling(bars_per_day * fast_days).std(ddof=0)
    med = roll_std.rolling(bars_per_day * slow_days).median().shift(1)
    return (roll_std.shift(1) > med).astype(float)


def apply_regime_mask(weights: pd.DataFrame, mask) -> pd.DataFrame:
    """Zero out positions on bars where the regime mask is False.
    `mask` can be a boolean series (global gate) or a DataFrame with the
    same shape as `weights` (per-stock gate).
    """
    if isinstance(mask, pd.DataFrame):
        m = mask.reindex_like(weights).fillna(0).astype(float)
        return weights.mul(m)
    m = mask.reindex(weights.index).fillna(False).astype(float)
    return weights.mul(m, axis=0)


def build_final_strategy(
    panel: pd.DataFrame,
    target_symbols: list[str],
    proxy_symbol: str = "XLK",
    signal_horizon: int = 30,
    ts_window: int = 390,
    z_in: float = 2.0,
    z_out_ratio: float = 0.2,
    hold_bars: int = 60,
    dispersion_fast: int = 5,
    dispersion_slow: int = 60,
    gross: float = 1.0,
) -> dict:
    """
    One-call constructor for the Notebook-03 final strategy: cross-sectional
    residual reversal with a single-channel dispersion regime gate.
    """
    wp = build_wide_panels(panel, target_symbols, proxy_symbol)
    ret = wp["ret_1m"]
    proxy = wp["proxy_ret_1m"]
    resid, beta = residualize(ret, proxy, window=ts_window)
    sig = resid.rolling(signal_horizon).sum()
    z = ts_zscore(sig, window=ts_window)
    w_raw = sparse_reversion_state_machine(
        z, z_in=z_in, z_out_ratio=z_out_ratio, hold_bars=hold_bars, gross=gross,
    )
    mask = dispersion_regime_filter(resid, dispersion_fast, dispersion_slow)
    w_final = apply_regime_mask(w_raw, mask)
    return {
        "weights_raw": w_raw,
        "weights_final": w_final,
        "returns_wide": ret,
        "residual_wide": resid,
        "beta_wide": beta,
        "zscore": z,
        "regime_mask": mask,
    }


def build_final_strategy_v2(
    panel: pd.DataFrame,
    target_symbols: list[str],
    proxy_symbol: str = "XLK",
    signal_horizon: int = 30,
    ts_window: int = 390,
    z_in: float = 2.0,
    z_out_ratio: float = 0.2,
    hold_bars: int = 60,
    dispersion_fast: int = 5,
    dispersion_slow: int = 60,
    use_breadth_gate: bool = True,
    use_per_stock_gate: bool = True,
    gross: float = 1.0,
) -> dict:
    """
    Phase-12 final strategy: cross-sectional residual reversal with a
    two-layer regime gate.

    Layer 1 (global)  = dispersion_regime_filter AND breadth_regime_filter
                        — only trade when residual dispersion is high AND
                        the dispersion is broad (not concentrated in one
                        name).
    Layer 2 (per-stock) = per_stock_vol_mask — only take positions in
                        stocks whose own residual vol is above their own
                        trailing median.

    Both layers are strictly lagged. Returns a superset of keys compared
    to build_final_strategy, including:
      regime_mask_disp      — Iteration E gate (baseline)
      regime_mask_broad     — breadth gate (Iteration F2 component)
      regime_mask_combined  — Iteration F2 combined global gate
      regime_mask_per_stock — Iteration G per-stock gate (DataFrame)
      weights_final         — after ALL enabled gates
    """
    wp = build_wide_panels(panel, target_symbols, proxy_symbol)
    ret = wp["ret_1m"]
    proxy = wp["proxy_ret_1m"]
    resid, beta = residualize(ret, proxy, window=ts_window)
    sig = resid.rolling(signal_horizon).sum()
    z = ts_zscore(sig, window=ts_window)
    w_raw = sparse_reversion_state_machine(
        z, z_in=z_in, z_out_ratio=z_out_ratio, hold_bars=hold_bars, gross=gross,
    )

    gates = breadth_regime_filter(resid, dispersion_fast, dispersion_slow)
    global_mask = gates["combined"] if use_breadth_gate else gates["disp_on"]
    w_after_global = apply_regime_mask(w_raw, global_mask)

    if use_per_stock_gate:
        psm = per_stock_vol_mask(resid, dispersion_fast, dispersion_slow)
        w_final = w_after_global.mul(psm.reindex_like(w_after_global).fillna(0))
    else:
        psm = None
        w_final = w_after_global

    return {
        "weights_raw": w_raw,
        "weights_after_global_gate": w_after_global,
        "weights_final": w_final,
        "returns_wide": ret,
        "residual_wide": resid,
        "beta_wide": beta,
        "zscore": z,
        "regime_mask_disp": gates["disp_on"],
        "regime_mask_broad": gates["broad_on"],
        "regime_mask_combined": gates["combined"],
        "regime_mask_per_stock": psm,
    }


def apply_vol_targeting(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_daily_vol: float = 0.01,
    vol_window: int = 390 * 5,
    leverage_cap: float = 3.0,
) -> pd.DataFrame:
    """Scale the whole book each bar so realised daily vol hits a target."""
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    realised = port_ret.rolling(vol_window, min_periods=vol_window // 5).std().shift(1)
    scale = (target_daily_vol / np.sqrt(390) / realised).clip(upper=leverage_cap)
    return weights.mul(scale, axis=0).fillna(0)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Execute the weights as target exposures.
      PnL[t]       = sum_s weights[t-1,s] * returns[t,s]       (no look-ahead)
      turnover[t]  = sum_s |weights[t,s] - weights[t-1,s]|
      cost[t]      = turnover[t] * cost_bps / 10000
      net_pnl[t]   = PnL[t] - cost[t]
    """
    w = weights.fillna(0).reindex_like(returns).fillna(0)
    r = returns.fillna(0)
    gross = (w.shift(1) * r).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0)
    cost = turnover * cost_bps / 10_000
    net = gross - cost
    equity = net.cumsum()
    return pd.DataFrame({
        "gross_pnl": gross,
        "turnover": turnover,
        "cost": cost,
        "net_pnl": net,
        "equity": equity,
        "gross_exposure": w.abs().sum(axis=1),
        "net_exposure": w.sum(axis=1),
    })


def portfolio_metrics(
    bt: pd.DataFrame,
    bars_per_day: int = 390,
    trading_days: int = 252,
) -> dict:
    """Core evaluation metrics for a portfolio PnL series."""
    pnl = bt["net_pnl"].values
    eq = bt["equity"].values
    if len(pnl) == 0:
        return {}
    ann_factor = np.sqrt(bars_per_day * trading_days)
    std = pnl.std(ddof=0)
    downside = pnl[pnl < 0]
    ds = downside.std(ddof=0) if len(downside) else 0.0
    sharpe = pnl.mean() / std * ann_factor if std > 0 else 0.0
    sortino = pnl.mean() / ds * ann_factor if ds > 0 else 0.0

    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    maxdd = dd.min() if len(dd) else 0.0

    n_days = len(pnl) / bars_per_day
    total_ret = eq[-1] if len(eq) else 0.0
    ann_ret = total_ret * (trading_days / max(n_days, 1))
    calmar = ann_ret / abs(maxdd) if maxdd < 0 else 0.0

    hit_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    # cost ratio: how much of gross was eaten by costs
    cost_ratio = bt["cost"].sum() / max(abs(bt["gross_pnl"].sum()), 1e-9)
    avg_gross = bt["gross_exposure"].mean()

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "ann_return": float(ann_ret),
        "total_return": float(total_ret),
        "max_drawdown": float(maxdd),
        "calmar": float(calmar),
        "bar_hit_rate": hit_rate,
        "avg_daily_turnover_x": float(bt["turnover"].mean() * bars_per_day / max(avg_gross, 1e-6)),
        "avg_gross_exposure": float(avg_gross),
        "cost_share_of_gross": float(cost_ratio),
        "n_bars": int(len(pnl)),
    }


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity - peak


# ---------------------------------------------------------------------------
# Meta-labelling
# ---------------------------------------------------------------------------

def triple_barrier_portfolio_labels(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 15,
    profit_k: float = 1.0,
    stop_k: float = 1.0,
    vol_scale: float = 0.005,
) -> pd.Series:
    """
    For each bar where at least one position exists, forward-simulate the PnL
    path for `horizon` bars. Label +1 if the *target* book hits
    +profit_k * vol_scale first, -1 if it hits -stop_k * vol_scale first,
    0 if neither.
    """
    pnl_path = (weights.shift(1) * returns).sum(axis=1)  # per-bar net returns
    is_active = (weights.abs().sum(axis=1) > 0).values
    pnl_arr = pnl_path.fillna(0).values
    n = len(pnl_arr)
    labels = np.zeros(n, dtype=int)
    upper = profit_k * vol_scale
    lower = -stop_k * vol_scale
    for i in range(n - 1):
        if not is_active[i]:
            continue
        end = min(n, i + horizon + 1)
        cum = np.cumsum(pnl_arr[i + 1:end])
        up = np.where(cum >= upper)[0]
        dn = np.where(cum <= lower)[0]
        if len(up) == 0 and len(dn) == 0:
            labels[i] = 0
        elif len(up) == 0:
            labels[i] = -1
        elif len(dn) == 0:
            labels[i] = 1
        else:
            labels[i] = 1 if up[0] < dn[0] else -1
    return pd.Series(labels, index=weights.index, dtype=int)


# ---------------------------------------------------------------------------
# Headline summary helpers
# ---------------------------------------------------------------------------

def format_metrics_row(name: str, m: dict) -> dict:
    def pct(x): return f"{x*100:+.2f}%" if x is not None and np.isfinite(x) else "—"
    def num(x, d=2): return f"{x:.{d}f}" if x is not None and np.isfinite(x) else "—"
    return {
        "version": name,
        "ann_return": pct(m.get("ann_return")),
        "sharpe": num(m.get("sharpe")),
        "sortino": num(m.get("sortino")),
        "max_dd": pct(m.get("max_drawdown")),
        "calmar": num(m.get("calmar")),
        "avg_gross": num(m.get("avg_gross_exposure"), 2),
        "daily_turnover": num(m.get("avg_daily_turnover_x"), 1) + "x",
        "cost_share": num(m.get("cost_share_of_gross"), 2),
        "bars": m.get("n_bars", 0),
    }
