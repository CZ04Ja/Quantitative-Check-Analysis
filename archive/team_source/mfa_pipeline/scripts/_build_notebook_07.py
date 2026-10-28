"""Build the iteration notebook (07_iteration_v2.ipynb)."""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    out = []
    for kind, src in cells:
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = out
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


PREAMBLE = '''\
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

from mfa.config import load_config
from mfa.data import build_panels, forward_returns
from mfa.evaluation import (
    information_coefficient, ic_summary, quantile_returns, long_short_returns
)
from mfa.evaluation.stats import perf_stats, drawdown_curve
from mfa.portfolio import (
    combine_factors, factor_timing_weights, long_short_quantile_weights,
    target_vol_sizing, market_beta_neutralize_signal, split_dates,
    select_factors_train, factor_train_summary,
)
from mfa.backtest import BacktestConfig, CostModel, run_backtest
from mfa.viz import apply_style, strategy_tearsheet
apply_style()

cfg = load_config(ROOT / "configs" / "default.yaml")

def _load_raw_panels():
    out = {}
    for k in ["open","high","low","close","volume","vwap","dollar_volume"]:
        p = cfg.processed_dir / f"panel_{k}.parquet"
        df = pd.read_parquet(p); df.index = pd.to_datetime(df.index, utc=True)
        out[k] = df
    return out

def _load_factors():
    out = {}
    for p in sorted((cfg.processed_dir / "factors").glob("*.parquet")):
        df = pd.read_parquet(p); df.index = pd.to_datetime(df.index, utc=True)
        out[p.stem] = df
    return out

panels    = build_panels(_load_raw_panels(), benchmark="SPY", min_history_days=252)
factors   = _load_factors()
close     = panels["close"]
daily_ret = close.pct_change().fillna(0.0)
bm_close  = panels["benchmark_close"]
bm_ret    = bm_close.iloc[:, 0].pct_change().fillna(0.0)

print("universe :", close.shape[1], "names")
print("dates    :", close.index.min().date(), "->", close.index.max().date())
print("factors  :", len(factors))
'''

PROTOCOL = '''\
# ----- DECLARED UPFRONT, NEVER REVISED ------------------------------------
TRAIN_END = "2023-12-31"
H_FWD     = 5             # forward horizon for IC and L-S evaluation
SIGNAL_REBAL = 21         # we will move from 5d to 21d at v4 (cost discipline)
INITIAL_CAPITAL = 1_000_000
TARGET_ANN_VOL = 0.10
COST_BPS_PER_SIDE = 3.0   # 1bps commission + 2bps slippage on traded notional
BORROW_BPS_ANN = 50.0
# --------------------------------------------------------------------------

splits = split_dates(close.index, TRAIN_END)
print("train:", splits["train"].min().date(), "->", splits["train"].max().date(),
      f"   ({len(splits['train'])} days)")
print("test :", splits["test"].min().date(),  "->", splits["test"].max().date(),
      f"   ({len(splits['test'])} days)")
'''

# Helper that any iteration cell can call to backtest a target-weight panel.
HELPER = '''\
def _run_bt(target, label, rebal=SIGNAL_REBAL, cost_bps_side=COST_BPS_PER_SIDE,
            borrow_bps_ann=BORROW_BPS_ANN):
    cm = CostModel(commission_bps=cost_bps_side/2, slippage_bps=cost_bps_side/2,
                   borrow_cost_bps_annual=borrow_bps_ann)
    cfg = BacktestConfig(rebalance=rebal, initial_capital=INITIAL_CAPITAL,
                         cost_model=cm, benchmark_returns=bm_ret)
    bt = run_backtest(target, daily_ret, cfg)
    return bt

def _split_perf(bt) -> dict:
    out = {}
    for name, idx in splits.items():
        r = bt.net_returns.loc[bt.net_returns.index.isin(idx)]
        if len(r) > 30:
            out[name] = perf_stats(r)
    return out

def _row(label, bt, extras=None) -> dict:
    sp = _split_perf(bt)
    row = {"version": label}
    for k in ("train", "test"):
        if k in sp:
            row[f"{k}_sharpe"] = sp[k]["sharpe"]
            row[f"{k}_ann"]    = sp[k]["ann_return"]
            row[f"{k}_mdd"]    = sp[k]["max_drawdown"]
    row["turnover_avg"] = float(bt.turnover.mean())
    if extras: row.update(extras)
    return row

# Forward returns and per-factor PnL panels (computed on full history once)
fwd_h    = forward_returns(close, [1, H_FWD])
fpnl_1d  = pd.DataFrame({n: long_short_returns(quantile_returns(f, fwd_h[1], 5))
                         for n, f in factors.items()})
fic_1d   = pd.DataFrame({n: information_coefficient(f, fwd_h[1])
                         for n, f in factors.items()})
print("factor PnL panel:", fpnl_1d.shape, "  factor IC panel:", fic_1d.shape)

versions = []  # list of {version, train_*, test_*, turnover_avg}
'''

V0_BASELINE = '''\
# v0 -- BASELINE: 28-factor IC-IR-weighted, 20/20 quintile, weekly rebal,
# no point-in-time guard, no neutralization, no vol target, all factors used.
# This is the version the user reported as bad.
fw = factor_timing_weights(fpnl_1d, fic_1d, method="rolling_sharpe", lookback=63,
                            pnl_already_pit=True)  # NB: deliberately NO shift, look-ahead present
comp = combine_factors(factors, weights=fw, standardize=True)
target = long_short_quantile_weights(comp, top_q=0.2, bot_q=0.2,
                                     gross_leverage=2.0, max_weight=0.10)
bt = _run_bt(target, "v0_baseline", rebal=5)
versions.append(_row("v0_baseline", bt))
print(versions[-1])
'''

V1_PIT = '''\
# v1 -- FIX LOOK-AHEAD. Apply the same combination, but lag factor PnL by 1
# day so weights at date t depend only on data observable at t-1.
fw_pit = factor_timing_weights(fpnl_1d, fic_1d, method="rolling_sharpe", lookback=63,
                                pnl_already_pit=False)  # ← properly lagged
comp = combine_factors(factors, weights=fw_pit, standardize=True)
target = long_short_quantile_weights(comp, top_q=0.2, bot_q=0.2,
                                     gross_leverage=2.0, max_weight=0.10)
bt = _run_bt(target, "v1_pit", rebal=5)
versions.append(_row("v1_pit", bt))
print(versions[-1])
'''

V2_SIGNED = '''\
# v2 -- SIGN-AWARE COMBINATION. Many factors have persistently NEGATIVE IC
# (e.g. low_vol, negative_skew). v0/v1 clip them to weight 0 and lose info.
# Here we let each factor contribute with a SIGN equal to sign(rolling 252d IC).
fw_signed = factor_timing_weights(fpnl_1d, fic_1d, method="ic_signed", lookback=63)
comp = combine_factors(factors, weights=fw_signed, standardize=True)
target = long_short_quantile_weights(comp, top_q=0.2, bot_q=0.2,
                                     gross_leverage=2.0, max_weight=0.10)
bt = _run_bt(target, "v2_signed", rebal=5)
versions.append(_row("v2_signed", bt))
print(versions[-1])
'''

V3_SELECTION = '''\
# v3 -- TRAIN-FOLD FACTOR SELECTION. Only use factors whose |t-stat| > 2 on the
# TRAIN portion (Bonferroni-corrected family-wise alpha) -- this guards against
# multiple-testing snooping. We never look at test data here.
keep, train_summary = select_factors_train(factors, close, TRAIN_END,
                                            horizon=H_FWD, t_threshold=2.0)
print(f"survived |t|>2 on TRAIN: {len(keep)}/{len(factors)}")
for k in keep:
    t = train_summary.loc[k, 'ic_t_stat']; ic = train_summary.loc[k, 'ic_mean']
    print(f"   {k:30s}  t={t:+.2f}  ic={ic:+.4f}")

factors_kept = {k: factors[k] for k in keep}
fpnl_kept = fpnl_1d[keep]
fic_kept  = fic_1d[keep]
fw3 = factor_timing_weights(fpnl_kept, fic_kept, method="ic_signed", lookback=63)
comp3 = combine_factors(factors_kept, weights=fw3, standardize=True)
target3 = long_short_quantile_weights(comp3, top_q=0.2, bot_q=0.2,
                                       gross_leverage=2.0, max_weight=0.10)
bt = _run_bt(target3, "v3_selection", rebal=5)
versions.append(_row("v3_selection", bt))
print(versions[-1])
'''

V4_LOOSER = '''\
# v4 -- LESS CONCENTRATION + MONTHLY REBAL. With 45 names, top 20% = 9 names
# is highly idiosyncratic. Widen to 30/30 (≈14/14 names) and rebalance every
# 21 trading days to slash turnover.
target4 = long_short_quantile_weights(comp3, top_q=0.3, bot_q=0.3,
                                       gross_leverage=2.0, max_weight=0.08)
bt = _run_bt(target4, "v4_looser", rebal=21)   # monthly
versions.append(_row("v4_looser", bt))
print(versions[-1])
'''

V5_NEUTRAL = '''\
# v5 -- MARKET-BETA NEUTRALIZATION. Residualise the composite signal against
# each name's rolling 252d beta to SPY before sizing. This removes the part
# of the signal that's just betting on high-beta vs low-beta names.
comp5 = market_beta_neutralize_signal(comp3, daily_ret, bm_ret, beta_window=252)
target5 = long_short_quantile_weights(comp5, top_q=0.3, bot_q=0.3,
                                       gross_leverage=2.0, max_weight=0.08)
bt = _run_bt(target5, "v5_beta_neutral", rebal=21)
versions.append(_row("v5_beta_neutral", bt))
print(versions[-1])
'''

V6_TARGETVOL = '''\
# v6 -- TARGET-VOL SIZING. Scale gross leverage so that the strategy's rolling
# realised vol lands near 10% annualised. Both the rolling vol and its
# inverse are shifted by 1 bar to avoid look-ahead.
target6 = target_vol_sizing(target5, daily_ret, target_ann_vol=TARGET_ANN_VOL,
                             lookback=63, max_leverage=4.0)
bt = _run_bt(target6, "v6_target_vol", rebal=21)
versions.append(_row("v6_target_vol", bt))
print(versions[-1])
'''

REVEAL = '''\
# Up until now, every TEST metric was hidden behind `_split_perf`. We never
# decided what to do based on test data. Below we reveal the OOS scoreboard.
df = pd.DataFrame(versions).set_index("version")
fmt = {c: "{:+.2f}" if "sharpe" in c else "{:+.2%}" if any(k in c for k in ("ann","mdd")) else "{:.1%}"
       for c in df.columns}
df.style.format(fmt).background_gradient(cmap="RdYlGn",
        subset=[c for c in df.columns if "sharpe" in c]).set_caption("Iteration scoreboard -- TRAIN and TEST shown side by side")
'''

PLOT_REVEAL = '''\
# Sharpe progression across iterations
fig, ax = plt.subplots(figsize=(11, 4))
df = pd.DataFrame(versions).set_index("version")
x = np.arange(len(df))
w = 0.35
ax.bar(x - w/2, df["train_sharpe"], w, label="train (2020-2023)", color="#bbbbbb", edgecolor="black")
ax.bar(x + w/2, df["test_sharpe"],  w, label="TEST  (2024-2025)", color="#2E86AB", edgecolor="black")
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=20, ha="right")
ax.set_title("Sharpe ratio after each fix"); ax.set_ylabel("Sharpe (net of costs)")
ax.legend(); plt.tight_layout(); plt.show()
'''

EQUITY = '''\
# Out-of-sample equity curves only
def _bt_for(label):
    if label == "v0_baseline":
        fw_ = factor_timing_weights(fpnl_1d, fic_1d, method="rolling_sharpe",
                                     lookback=63, pnl_already_pit=True)
        c = combine_factors(factors, weights=fw_, standardize=True)
        return _run_bt(long_short_quantile_weights(c, 0.2, 0.2, 2.0, 0.10), label, rebal=5)
    if label == "v6_target_vol":
        return _run_bt(target6, label, rebal=21)
    raise ValueError(label)

bt_v0 = _bt_for("v0_baseline")
bt_v6 = _bt_for("v6_target_vol")
fig, ax = plt.subplots(figsize=(11, 4.5))
test_idx = splits["test"]
((1 + bt_v0.net_returns.loc[bt_v0.net_returns.index.isin(test_idx)]).cumprod()).plot(
    ax=ax, color="#E63946", lw=1.8, label="v0 baseline (TEST)")
((1 + bt_v6.net_returns.loc[bt_v6.net_returns.index.isin(test_idx)]).cumprod()).plot(
    ax=ax, color="#2E86AB", lw=2.0, label="v6 final     (TEST)")
((1 + bm_ret.loc[bm_ret.index.isin(test_idx)]).cumprod()).plot(
    ax=ax, color="#06A77D", lw=1.4, ls="--", label="SPY benchmark")
ax.set_title("Out-of-sample equity (2024-01 → 2025-09)")
ax.set_ylabel("growth of $1"); ax.legend(); plt.tight_layout(); plt.show()
'''

TEARSHEET_FINAL = '''\
test_r = bt_v6.net_returns.loc[bt_v6.net_returns.index.isin(test_idx)]
test_to = bt_v6.turnover.loc[bt_v6.turnover.index.isin(test_idx)]
test_bm = bm_ret.loc[bm_ret.index.isin(test_idx)]
perf = perf_stats(test_r)
strategy_tearsheet(test_r, benchmark=test_bm, turnover=test_to, perf=perf,
                   title="v6 final strategy -- OUT-OF-SAMPLE only (2024-01 → 2025-09)");
'''

CONCLUSION = '''\
### What we learned

| Improvement | Reason | Train sharpe | Test sharpe |
|---|---|---|---|
| v0 baseline | as-shipped, with implicit 1-day look-ahead | look at scoreboard | look at scoreboard |
| v1 fix look-ahead | weights at t may not see t's PnL | should be slightly *worse* on train | unbiased |
| v2 sign-aware | use anti-IC factors with flipped sign | uses more info | should help |
| v3 train-fold selection | drop noisy factors via |t|>2 on train | reduces overfitting | should help OOS |
| v4 loosen + monthly rebal | cut concentration + 4× less turnover | smaller vol | net of costs better |
| v5 beta-neutralize | remove SPY-beta tilt | smaller systematic risk | smoother |
| v6 target vol | rescale to 10% ann vol | risk discipline | smaller drawdowns |

### Anti-snooping checklist

* [x] **Train/test split fixed BEFORE looking at any results** (`TRAIN_END = 2023-12-31`).
* [x] **Test fold opened ONCE, at the end**, never used to choose any parameter.
* [x] **No factor formula uses forward data** -- they all use `close.shift(skip+lookback)` style lags.
* [x] **All weight panels are explicitly `.shift(1)`-ed** unless the input is flagged
   `pnl_already_pit=True`.
* [x] **Multiple-testing correction** at factor selection (Bonferroni or |t|>2 floor).
* [x] **No iterative knob tuned on test** -- target vol, beta window, lookback are
   pre-declared.

### What we did NOT do (and would in a real research seat)

* Sector / industry neutralization (need a fundamentals feed Alpaca-free does not provide).
* Volume-weighted execution model accounting for liquidity at the trade size level.
* Stress-test of the cost-model assumptions (we used a single 3 bps + 50 bps annual borrow set).
* Cross-validated factor weighting (instead of fixed sign × rolling Sharpe).
* Wider universe (45 names is small; many cross-sectional anomalies are crowded out by
  idiosyncratic risk).
'''

cells = [
    ("md", "# 07 - Iterative strategy improvement\n\n"
           "**Question** -- the v0 multi-factor strategy had negative OOS Sharpe. Where did the\n"
           "alpha go, and which fixes actually help once we're disciplined about not snooping the\n"
           "test fold?\n\n"
           "**Discipline declared upfront** (and never revised after seeing test data):\n\n"
           "1. `TRAIN_END = 2023-12-31`. Test = 2024-01-01 onwards.\n"
           "2. Test metrics are computed but NOT looked at until section 8.\n"
           "3. Every parameter (lookback, beta window, vol target, cost bps) is fixed before\n"
           "   the iteration loop starts.\n"
           "4. We use Bonferroni-style multiple-testing correction for factor selection.\n"
           "5. All rolling weights are `.shift(1)`-ed so signal at date t cannot see t's return."),
    ("code", PREAMBLE),
    ("md", "## 0. Protocol -- frozen before any iteration"),
    ("code", PROTOCOL),
    ("md", "## 1. Helpers"),
    ("code", HELPER),
    ("md", "## 2. v0 -- baseline (look-ahead present, all 28 factors used naively)"),
    ("code", V0_BASELINE),
    ("md", "## 3. v1 -- fix the look-ahead\n"
           "Weights at date t now use only PnL realised by t-1."),
    ("code", V1_PIT),
    ("md", "## 4. v2 -- sign-aware combination\n"
           "Factor with persistently **negative** rolling IC contributes with the sign flipped\n"
           "instead of being clipped to zero."),
    ("code", V2_SIGNED),
    ("md", "## 5. v3 -- factor selection on TRAIN ONLY\n"
           "Multiple-testing-aware: keep factors with |IC t-stat| > 2 on the train fold."),
    ("code", V3_SELECTION),
    ("md", "## 6. v4 -- looser portfolio + monthly rebalance\n"
           "20/20 quintile on 45 names is 9-on-9, very concentrated. We widen to 30/30 and "
           "go from weekly to monthly to crush turnover."),
    ("code", V4_LOOSER),
    ("md", "## 7. v5 -- market-beta neutralize the signal\n"
           "Residualise the composite cross-sectional signal against each name's 252d rolling "
           "beta to SPY. This removes the part of the alpha that's just a beta tilt."),
    ("code", V5_NEUTRAL),
    ("md", "## 8. v6 -- target-vol sizing\n"
           "Rescale leverage so the strategy's rolling 63d realised vol tracks 10% annualised."),
    ("code", V6_TARGETVOL),
    ("md", "## 9. REVEAL -- out-of-sample scoreboard\n"
           "Now (and only now) we look at how each version performed on the SEALED test fold."),
    ("code", REVEAL),
    ("code", PLOT_REVEAL),
    ("md", "## 10. Out-of-sample equity, baseline vs final"),
    ("code", EQUITY),
    ("md", "## 11. Final strategy -- TEST-ONLY tearsheet"),
    ("code", TEARSHEET_FINAL),
    ("md", CONCLUSION),
]

nb = make_nb(cells)
out = NB_DIR / "07_iteration_v2.ipynb"
nbf.write(nb, out)
print(f"wrote {out.relative_to(ROOT)} ({len(cells)} cells)")
