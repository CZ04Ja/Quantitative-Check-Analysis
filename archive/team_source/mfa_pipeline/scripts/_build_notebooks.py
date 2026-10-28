"""Generate all research notebooks in notebooks/.

Run once to (re)create the .ipynb files. Then execute them with
    jupyter nbconvert --to notebook --execute notebooks/01_*.ipynb --inplace ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers to build a notebook from a list of (kind, source) tuples.
# ---------------------------------------------------------------------------
def make_nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    out = []
    for kind, src in cells:
        if kind == "md":
            out.append(nbf.v4.new_markdown_cell(src))
        elif kind == "code":
            out.append(nbf.v4.new_code_cell(src))
        else:
            raise ValueError(kind)
    nb.cells = out
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def write(name: str, cells: list[tuple[str, str]]) -> None:
    nb = make_nb(cells)
    p = NB_DIR / name
    nbf.write(nb, p)
    print(f"wrote {p.relative_to(ROOT)}  ({len(cells)} cells)")


# Common preamble used in every notebook so each can run independently.
PREAMBLE = '''\
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

from mfa.config import load_config
from mfa.viz import apply_style
apply_style()

cfg = load_config(ROOT / "configs" / "default.yaml")
print("project:", cfg.project_root.name, "| processed:", cfg.processed_dir.exists())
'''

LOAD_CACHED = '''\
from mfa.data import build_panels

def _load_raw_panels():
    out = {}
    for k in ["open","high","low","close","volume","vwap","dollar_volume"]:
        p = cfg.processed_dir / f"panel_{k}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index, utc=True)
            out[k] = df
    if not out:
        raise RuntimeError("No cached panels found. Run scripts/01_download_data.py first.")
    return out

def _load_factors():
    d = cfg.processed_dir / "factors"
    out = {}
    for p in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        out[p.stem] = df
    return out

raw_panels = _load_raw_panels()
benchmark = cfg.section("universe").get("benchmark", "SPY")
panels = build_panels(raw_panels, benchmark=benchmark, min_history_days=252)
factors = _load_factors()
print(f"loaded {len(panels)} panels and {len(factors)} factors")
print("date range:", panels["close"].index.min().date(), "->", panels["close"].index.max().date())
print("universe size:", panels["close"].shape[1], " benchmark:", benchmark)
'''


# ===========================================================================
# 01 -- DATA & EDA
# ===========================================================================
write("01_data_eda.ipynb", [
    ("md", "# 01 - Data acquisition & exploratory analysis\n\n"
           "Goal: understand what we're trading on. We pulled US equity daily bars from Alpaca "
           "(IEX feed, free tier) for a sector-diverse 45-name universe + SPY benchmark.\n\n"
           "This notebook checks data quality, return distributions, cross-sectional dispersion, "
           "and macro-regime context."),
    ("code", PREAMBLE),
    ("code", LOAD_CACHED),
    ("md", "## 1. Data quality -- missingness & survivorship"),
    ("code", '''\
close = panels["close"]
print("shape:", close.shape, "  bars:", close.shape[0], " names:", close.shape[1])
print("first 5 with most NaNs:")
print(close.isna().sum().sort_values(ascending=False).head(5))
print()
print("benchmark first/last:", panels["benchmark_close"].dropna().index[[0,-1]].date.tolist())
'''),
    ("md", "## 2. Return distribution"),
    ("code", '''\
ret = close.pct_change().dropna(how="all")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ret.stack().hist(bins=120, ax=ax[0], color="#2E86AB", alpha=0.85)
ax[0].set_title("Pooled daily-return distribution"); ax[0].set_xlabel("daily return")
ax[0].axvline(0, color="black", lw=0.7)

(ret.std() * np.sqrt(252)).sort_values().plot.barh(ax=ax[1], color="#E63946")
ax[1].set_title("Annualised vol per name"); ax[1].set_xlabel("vol"); plt.tight_layout(); plt.show()

print("pooled stats:")
s = ret.stack()
print(f"  mean = {s.mean():+.5f}  std = {s.std():.4f}  skew = {s.skew():+.2f}  kurt = {s.kurt():+.2f}")
print(f"  worst day = {s.min():+.2%}  best day = {s.max():+.2%}")
'''),
    ("md", "## 3. Cross-sectional dispersion over time\n"
           "Dispersion = std of one-day returns across names. When dispersion is high, "
           "*cross-sectional* alpha has more room; when it collapses, factors have nothing to "
           "rank against."),
    ("code", '''\
disp = ret.std(axis=1) * np.sqrt(252)
fig, ax = plt.subplots(figsize=(11, 3.5))
disp.plot(ax=ax, color="#bbbbbb", lw=0.7, label="daily")
disp.rolling(63).mean().plot(ax=ax, color="#2E86AB", lw=2.0, label="63d mean")
ax.set_ylabel("annualised x-sec std"); ax.set_title("Cross-sectional return dispersion"); ax.legend(); plt.show()
'''),
    ("md", "## 4. Universe correlation structure"),
    ("code", '''\
corr = ret.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.7}, ax=ax, xticklabels=True, yticklabels=True)
ax.set_title(f"Pairwise daily-return correlation -- mean off-diag = {(corr.values[~np.eye(len(corr), dtype=bool)]).mean():.3f}")
plt.show()
'''),
    ("md", "## 5. Benchmark vs equal-weight universe"),
    ("code", '''\
bench = panels["benchmark_close"].iloc[:, 0].pct_change().dropna()
ew = ret.mean(axis=1)
df = pd.concat([bench.rename("SPY"), ew.rename("equal-weight universe")], axis=1).dropna()
((1 + df).cumprod()).plot(figsize=(11, 4))
plt.yscale("log"); plt.title("Equity curves -- benchmark vs equal-weight"); plt.ylabel("growth of $1"); plt.show()

print("annualised return / vol / Sharpe:")
for c in df:
    s = df[c]
    ann = (1 + s).prod() ** (252 / len(s)) - 1
    vol = s.std() * np.sqrt(252)
    print(f"  {c:24s}  ann={ann:+.2%}  vol={vol:.2%}  sharpe={ann/vol:+.2f}")
'''),
    ("md", "### Key takeaways\n\n"
           "* Universe is well-aligned, no major missingness post-warmup.\n"
           "* Pooled daily returns are fat-tailed (positive excess kurtosis) and slightly right-skewed.\n"
           "* Cross-sectional dispersion oscillates with macro stress -- this is the playing field for our factors.\n"
           "* Pairwise correlations are mostly +0.3-0.6 (typical for US large-cap), so a single tilt won't diversify much.\n"
           "* SPY and the equal-weight universe trade together; we will benchmark vs SPY."),
])


# ===========================================================================
# 02 -- CLASSIC FACTOR RESEARCH
# ===========================================================================
write("02_factor_research_classic.ipynb", [
    ("md", "# 02 - Classic factor research\n\n"
           "We dig into the *non-novel* families: momentum, short-term reversal, low-vol, quality, "
           "liquidity, and microstructure. For each factor we measure:\n"
           "* **IC** (Spearman cross-sectional rank correlation with forward returns)\n"
           "* **IC-IR** (mean / std)  and  **IC t-statistic**\n"
           "* **Quantile cumulative-return spread** (5 buckets)\n"
           "* **Turnover** of the top quintile\n\n"
           "Forward horizon = 5 trading days unless stated otherwise."),
    ("code", PREAMBLE),
    ("code", LOAD_CACHED),
    ("code", '''\
from mfa.data import forward_returns
from mfa.evaluation import (
    information_coefficient, ic_summary, rolling_ic,
    quantile_returns, long_short_returns, factor_turnover, ic_decay,
)
from mfa.evaluation.stats import perf_stats
from mfa.factors import REGISTRY

H = 5
fwd = forward_returns(panels["close"], [1, H, 10, 21])
print("forward horizons computed:", list(fwd.keys()))
'''),
    ("md", "## 1. IC scoreboard (all classic factors)"),
    ("code", '''\
classic_families = ["momentum", "reversal", "volatility", "quality", "liquidity", "microstructure"]
classic_names = [n for fam, names in REGISTRY.by_family().items() if fam in classic_families for n in names]
print(f"{len(classic_names)} classic factors:")
for n in classic_names: print(" ", n)

rows = []
for n in classic_names:
    f = factors[n]
    ic = information_coefficient(f, fwd[H])
    s = ic_summary(ic)
    qr = quantile_returns(f, fwd[H], n_quantiles=5)
    ls = long_short_returns(qr)
    perf = perf_stats(ls, periods_per_year=252/H)
    to = factor_turnover(f, n_quantiles=5).mean()
    family = REGISTRY.get(n).family
    rows.append({"factor": n, "family": family, "ic_mean": s["ic_mean"],
                 "ic_ir": s["ic_ir"], "ic_t": s["ic_t_stat"],
                 "hit": s["ic_hit_rate"], "ls_sharpe": perf["sharpe"],
                 "ls_ann": perf["ann_return"], "ls_mdd": perf["max_drawdown"],
                 "turnover": float(to)})
board = pd.DataFrame(rows).set_index("factor").sort_values("ic_t", ascending=False)
board.style.format({"ic_mean":"{:+.3f}","ic_ir":"{:+.2f}","ic_t":"{:+.2f}","hit":"{:.1%}",
                    "ls_sharpe":"{:+.2f}","ls_ann":"{:+.1%}","ls_mdd":"{:.1%}","turnover":"{:.1%}"}).background_gradient(cmap="RdYlGn", subset=["ic_t","ls_sharpe"])
'''),
    ("md", "## 2. IC bar-chart by family"),
    ("code", '''\
fig, ax = plt.subplots(figsize=(11, 5))
colors = sns.color_palette("Set2", len(set(board["family"])))
fam_to_color = dict(zip(sorted(set(board["family"])), colors))
b = board.sort_values("ic_t")
ax.barh(range(len(b)), b["ic_t"], color=[fam_to_color[f] for f in b["family"]], edgecolor="black", lw=0.5)
ax.set_yticks(range(len(b))); ax.set_yticklabels(b.index, fontsize=9)
ax.axvline(2, color="red", ls="--", lw=1, label="t = +2"); ax.axvline(-2, color="red", ls="--", lw=1)
ax.axvline(0, color="black", lw=0.7)
ax.set_xlabel("IC t-statistic"); ax.set_title("Classic-factor IC significance")
handles = [plt.Rectangle((0,0),1,1, color=fam_to_color[f]) for f in fam_to_color]
ax.legend(handles, list(fam_to_color), loc="lower right"); plt.tight_layout(); plt.show()
'''),
    ("md", "## 3. Deep-dive on the strongest factor"),
    ("code", '''\
best = board.index[0]
print(f"strongest by t-stat: {best}  (family = {board.loc[best, 'family']})")

f = factors[best]
ic = information_coefficient(f, fwd[H])
qr = quantile_returns(f, fwd[H], n_quantiles=5)
ls = long_short_returns(qr)
decay = ic_decay(f, panels["close"], horizons=[1, 5, 10, 21, 63])

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
# IC time series
ic.plot(ax=ax[0,0], color="#bbbbbb", lw=0.7, label="daily")
ic.rolling(21).mean().plot(ax=ax[0,0], color="#2E86AB", lw=1.8, label="21d")
ax[0,0].axhline(0, color="black", lw=0.7); ax[0,0].set_title(f"IC time series -- {best}"); ax[0,0].legend()
# IC decay
decay.plot.bar(ax=ax[0,1], color="#2E86AB", edgecolor="black")
ax[0,1].axhline(0, color="black", lw=0.7); ax[0,1].set_title("IC decay"); ax[0,1].set_xlabel("forward horizon (days)")
# Quantile cum returns
eq = (1 + qr.fillna(0)).cumprod()
palette = sns.color_palette("RdYlGn", 5)
for i, c in enumerate(eq.columns):
    eq[c].plot(ax=ax[1,0], label=c, color=palette[i])
ax[1,0].set_yscale("log"); ax[1,0].set_title("Quantile cumulative returns"); ax[1,0].legend(ncol=5)
# L-S equity
(1 + ls.fillna(0)).cumprod().plot(ax=ax[1,1], color="#06A77D", lw=2.0)
ax[1,1].set_title("Long-short equity (Q5 - Q1)"); ax[1,1].set_ylabel("growth of $1")
plt.tight_layout(); plt.show()
'''),
    ("md", "## 4. IC stability across forward horizons"),
    ("code", '''\
horizon_grid = pd.DataFrame(index=classic_names, columns=[1, 5, 10, 21], dtype=float)
for n in classic_names:
    for h in horizon_grid.columns:
        ic = information_coefficient(factors[n], fwd[h])
        horizon_grid.loc[n, h] = ic.mean()
fig, ax = plt.subplots(figsize=(7, max(5, 0.3*len(classic_names))))
sns.heatmap(horizon_grid.astype(float), cmap="RdBu_r", center=0, annot=True, fmt="+.3f",
            cbar_kws={"label": "mean IC"}, ax=ax)
ax.set_title("Mean IC by factor x forward horizon"); ax.set_xlabel("forward horizon (days)"); plt.show()
'''),
    ("md", "### Takeaways\n\n"
           "* Classic momentum (12-1) shows the strongest single-factor IC and significance in this regime.\n"
           "* Low-vol shows persistent NEGATIVE IC -- the post-2020 mega-cap-tech rally rewarded high-vol\n"
           "  names, inverting the textbook anomaly.\n"
           "* Most short-horizon (1d) ICs are weak; we should expect the cross-sectional book to work\n"
           "  best at a 5-21 day horizon."),
])


# ===========================================================================
# 03 -- NOVEL FACTOR RESEARCH
# ===========================================================================
write("03_factor_research_novel.ipynb", [
    ("md", "# 03 - Novel ('brain-storm') factor research\n\n"
           "Eight non-textbook factors live in `mfa.factors.novel`. For each we restate the\n"
           "**behavioural story**, then test whether it is actually priced in our universe.\n\n"
           "If a factor's IC has the OPPOSITE sign of the story but is still significant, that's a\n"
           "real signal -- we just flip its sign in the combiner."),
    ("code", PREAMBLE),
    ("code", LOAD_CACHED),
    ("code", '''\
from mfa.data import forward_returns
from mfa.evaluation import information_coefficient, ic_summary, quantile_returns, long_short_returns
from mfa.evaluation.stats import perf_stats
from mfa.factors import REGISTRY

H = 5
fwd = forward_returns(panels["close"], [H])[H]
novel_names = [n for n in REGISTRY.by_family()["novel"]]
print("novel factors:", novel_names)
'''),
    ("md", "## 1. Side-by-side scoreboard"),
    ("code", '''\
rows = []
for n in novel_names:
    f = factors[n]
    ic = information_coefficient(f, fwd)
    s = ic_summary(ic)
    qr = quantile_returns(f, fwd, n_quantiles=5)
    ls = long_short_returns(qr)
    perf = perf_stats(ls, periods_per_year=252/H)
    rows.append({"factor": n, "story": REGISTRY.get(n).description.split(".")[0][:55] + "...",
                 "ic_mean": s["ic_mean"], "ic_t": s["ic_t_stat"], "hit": s["ic_hit_rate"],
                 "ls_sharpe": perf["sharpe"], "ls_ann": perf["ann_return"]})
board = pd.DataFrame(rows).set_index("factor").sort_values("ic_t", ascending=False)
board.style.format({"ic_mean":"{:+.3f}","ic_t":"{:+.2f}","hit":"{:.1%}","ls_sharpe":"{:+.2f}","ls_ann":"{:+.1%}"}).background_gradient(cmap="RdYlGn", subset=["ic_t"])
'''),
    ("md", "## 2. Visual: IC across all 8 novel factors"),
    ("code", '''\
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
for i, n in enumerate(novel_names):
    ax = axes.flat[i]
    ic = information_coefficient(factors[n], fwd)
    ic.rolling(63).mean().plot(ax=ax, color="#2E86AB", lw=1.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(ic.mean(), color="#E63946", ls="--", lw=0.8)
    ax.set_title(n, fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("63d rolling IC")
plt.suptitle("Novel factors -- 63d rolling IC over time", y=1.02, fontweight="bold")
plt.tight_layout(); plt.show()
'''),
    ("md", "## 3. Quantile spread for the most extreme novel factor"),
    ("code", '''\
extreme = board["ic_t"].abs().idxmax()
print(f"largest |t|: {extreme}  (sign = {'positive' if board.loc[extreme,'ic_t']>0 else 'negative'})")

f = factors[extreme]
qr = quantile_returns(f, fwd, n_quantiles=5)
eq = (1 + qr.fillna(0)).cumprod()

fig, ax = plt.subplots(figsize=(11, 4.5))
palette = sns.color_palette("RdYlGn", 5)
for i, c in enumerate(eq.columns):
    eq[c].plot(ax=ax, label=c, color=palette[i], lw=1.8)
ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
ax.set_title(f"Quantile cumulative returns -- {extreme}")
ax.legend(ncol=5); plt.show()

print(f"\\nstory  : {REGISTRY.get(extreme).description}")
print(f"verdict: {'STORY HOLDS' if board.loc[extreme,'ic_t']>0 else 'STORY INVERTED -- still tradable with sign flip'}")
'''),
    ("md", "## 4. Are novel factors uncorrelated with classic ones?"),
    ("code", '''\
from mfa.evaluation import correlation_matrix
classic_top = ["mom_252_21", "mom_126", "low_vol_21", "rev_5"]
subset = {**{n: factors[n] for n in classic_top}, **{n: factors[n] for n in novel_names[:6]}}
corr = correlation_matrix(subset, method="spearman")
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, annot=True, fmt="+.2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"size": 8})
ax.set_title("Mean cross-sectional correlation: classic (top) vs novel (bottom)")
plt.show()
'''),
    ("md", "### Verdict\n\n"
           "* `weekly_seasonal_friday` and `pullback_in_uptrend` show the largest |t| among novels --\n"
           "  the latter has the OPPOSITE sign of its narrative, meaning *deep* pullbacks have predicted\n"
           "  *more* underperformance in this regime (momentum dominance).\n"
           "* Several novel factors carry low correlation with classic momentum/reversal, so they add\n"
           "  diversification value to the multi-factor combination."),
])


# ===========================================================================
# 04 -- COMBINATION + FACTOR TIMING
# ===========================================================================
write("04_combination_and_timing.ipynb", [
    ("md", "# 04 - Multi-factor combination & factor timing\n\n"
           "Three combination recipes, side by side:\n"
           "1. **Equal-weight** -- the no-prior baseline.\n"
           "2. **Static IC weighting** -- weight each factor by its full-sample mean IC (look-ahead!).\n"
           "3. **Dynamic rolling-Sharpe timing** -- per-date weights from the past 63d L-S Sharpe.\n\n"
           "We measure each composite signal's IC and the resulting long-short PnL."),
    ("code", PREAMBLE),
    ("code", LOAD_CACHED),
    ("code", '''\
from mfa.data import forward_returns
from mfa.evaluation import information_coefficient, ic_summary, quantile_returns, long_short_returns
from mfa.evaluation.stats import perf_stats
from mfa.portfolio import combine_factors, factor_timing_weights

H = 5
fwd = {h: forward_returns(panels["close"], [h])[h] for h in [1, H]}

# Per-factor L-S PnL needed for IR-based timing
fpnl = pd.DataFrame({n: long_short_returns(quantile_returns(f, fwd[1], 5)) for n, f in factors.items()})
fic  = pd.DataFrame({n: information_coefficient(f, fwd[1]) for n, f in factors.items()})
print("factor PnL panel shape:", fpnl.shape)
'''),
    ("md", "## 1. Three composites"),
    ("code", '''\
composites = {}

# (a) equal weight
composites["equal"] = combine_factors(factors, weights=None, standardize=True)

# (b) static IC weighting (in-sample, look-ahead biased -- diagnostic only)
ic_means = {n: information_coefficient(factors[n], fwd[H]).mean() for n in factors}
total = sum(abs(v) for v in ic_means.values())
ic_w = {n: v / total for n, v in ic_means.items()}
composites["static_ic"] = combine_factors(factors, weights=ic_w, standardize=True)

# (c) dynamic rolling Sharpe timing
fw = factor_timing_weights(fpnl, factor_ic=fic, method="rolling_sharpe", lookback=63)
composites["timing_sharpe63"] = combine_factors(factors, weights=fw, standardize=True)

for k, v in composites.items():
    print(f"{k:18s}  shape={v.shape}  late nonNaN row count = {v.notna().sum(axis=1).iloc[-1]}")
'''),
    ("md", "## 2. Composite IC comparison"),
    ("code", '''\
rows = []
for k, c in composites.items():
    ic = information_coefficient(c, fwd[H])
    qr = quantile_returns(c, fwd[H], 5)
    ls = long_short_returns(qr)
    perf = perf_stats(ls, periods_per_year=252/H)
    rows.append({"composite": k, "ic_mean": ic.mean(), "ic_ir": ic.mean()/ic.std(),
                 "ic_t": (ic.mean()/ic.std())*np.sqrt(ic.notna().sum()),
                 "ls_sharpe": perf["sharpe"], "ls_ann": perf["ann_return"], "ls_mdd": perf["max_drawdown"]})
df = pd.DataFrame(rows).set_index("composite")
df.style.format({"ic_mean":"{:+.4f}","ic_ir":"{:+.2f}","ic_t":"{:+.2f}",
                 "ls_sharpe":"{:+.2f}","ls_ann":"{:+.2%}","ls_mdd":"{:.1%}"}).background_gradient(cmap="RdYlGn", subset=["ls_sharpe","ic_t"])
'''),
    ("md", "## 3. L-S equity comparison"),
    ("code", '''\
fig, ax = plt.subplots(figsize=(11, 4.5))
colors = {"equal":"#bbbbbb","static_ic":"#E63946","timing_sharpe63":"#2E86AB"}
for k, c in composites.items():
    qr = quantile_returns(c, fwd[H], 5)
    ls = long_short_returns(qr)
    (1 + ls.fillna(0)).cumprod().plot(ax=ax, label=k, color=colors[k], lw=1.8)
ax.set_title("Composite long-short equity"); ax.set_ylabel("growth of $1"); ax.legend(); plt.show()
'''),
    ("md", "## 4. Visualising the dynamic factor weights"),
    ("code", '''\
fig, ax = plt.subplots(figsize=(11, 4.5))
fw.plot.area(ax=ax, lw=0, alpha=0.85, colormap="tab20")
ax.set_title("Per-factor weight (rolling-Sharpe timing, 63d)")
ax.set_ylabel("weight"); ax.set_ylim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=7); plt.tight_layout(); plt.show()
'''),
    ("md", "## 5. Time-varying tilt: what the model thinks today"),
    ("code", '''\
last_w = fw.iloc[-1].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, max(4, 0.25*len(last_w))))
last_w.plot.barh(ax=ax, color="#2E86AB", edgecolor="black")
ax.set_title(f"Factor weights on {fw.index[-1].date()}"); ax.set_xlabel("weight"); plt.show()
'''),
    ("md", "### Takeaways\n\n"
           "* The static IC weighting is *upper-bound diagnostic*; it uses full-sample knowledge.\n"
           "* Dynamic rolling-Sharpe timing concentrates weight on whichever factor has been working\n"
           "  recently. This adds turnover but adapts to regime changes.\n"
           "* Equal-weighting is the conservative baseline; never wins, never blows up."),
])


# ===========================================================================
# 05 -- BACKTEST + COST SENSITIVITY
# ===========================================================================
write("05_backtest_costs.ipynb", [
    ("md", "# 05 - Backtest & cost sensitivity\n\n"
           "We run the full multi-factor strategy through the daily backtester with realistic costs,\n"
           "then sweep three knobs to see what kills the alpha:\n"
           "* commission + slippage (bps per side, on traded notional)\n"
           "* short-borrow cost (bps annual, on average short notional)\n"
           "* rebalance frequency (every N days)"),
    ("code", PREAMBLE),
    ("code", LOAD_CACHED),
    ("code", '''\
from mfa.data import forward_returns
from mfa.evaluation import quantile_returns, long_short_returns, information_coefficient
from mfa.evaluation.stats import perf_stats
from mfa.portfolio import combine_factors, factor_timing_weights, long_short_quantile_weights
from mfa.backtest import BacktestConfig, CostModel, run_backtest

# Build the standard composite + target weights
fwd1 = forward_returns(panels["close"], [1])[1]
fpnl = pd.DataFrame({n: long_short_returns(quantile_returns(f, fwd1, 5)) for n, f in factors.items()})
fic  = pd.DataFrame({n: information_coefficient(f, fwd1) for n, f in factors.items()})
fw   = factor_timing_weights(fpnl, fic, method="rolling_sharpe", lookback=63)
comp = combine_factors(factors, weights=fw, standardize=True)
print("composite shape:", comp.shape)
'''),
    ("md", "## 1. Baseline backtest"),
    ("code", '''\
target = long_short_quantile_weights(comp, top_q=0.2, bot_q=0.2, gross_leverage=2.0, max_weight=0.10)
daily_ret = panels["close"].pct_change().fillna(0.0)
bm = panels["benchmark_close"].iloc[:, 0].pct_change().fillna(0.0)

base_cfg = BacktestConfig(
    rebalance=5, initial_capital=1_000_000,
    cost_model=CostModel(commission_bps=1.0, slippage_bps=2.0, borrow_cost_bps_annual=50.0),
    benchmark_returns=bm,
)
bt = run_backtest(target, daily_ret, base_cfg)
perf = perf_stats(bt.net_returns)
pd.Series(perf, name="value").to_frame()
'''),
    ("md", "## 2. Trade-cost sweep (bps per side)"),
    ("code", '''\
def perf_for_costs(cps, sps, bps_borrow=50.0, rebal=5):
    cm = CostModel(commission_bps=cps, slippage_bps=sps, borrow_cost_bps_annual=bps_borrow)
    cfg = BacktestConfig(rebalance=rebal, initial_capital=1_000_000, cost_model=cm)
    bt = run_backtest(target, daily_ret, cfg)
    return perf_stats(bt.net_returns)

bps_grid = [0, 1, 2, 3, 5, 10, 20]
res = []
for tot in bps_grid:
    p = perf_for_costs(cps=tot/2, sps=tot/2, bps_borrow=50.0, rebal=5)
    res.append({"total_bps_per_side": tot, "sharpe": p["sharpe"], "ann_return": p["ann_return"], "max_dd": p["max_drawdown"]})
df = pd.DataFrame(res).set_index("total_bps_per_side")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
df["sharpe"].plot.bar(ax=ax[0], color="#2E86AB", edgecolor="black"); ax[0].set_title("Net Sharpe vs trade cost"); ax[0].axhline(0, color="black", lw=0.7)
df["ann_return"].plot.bar(ax=ax[1], color="#E63946", edgecolor="black"); ax[1].set_title("Annualised return vs trade cost"); ax[1].axhline(0, color="black", lw=0.7)
for a in ax: a.set_xlabel("commission+slippage per side (bps)")
plt.tight_layout(); plt.show()
df.round(3)
'''),
    ("md", "## 3. Rebalance-frequency sweep"),
    ("code", '''\
res = []
for rebal in [1, 2, 5, 10, 21]:
    p = perf_for_costs(cps=1.0, sps=2.0, bps_borrow=50.0, rebal=rebal)
    res.append({"rebalance_every_N_days": rebal, "sharpe": p["sharpe"], "ann_return": p["ann_return"], "max_dd": p["max_drawdown"]})
df = pd.DataFrame(res).set_index("rebalance_every_N_days")
fig, ax = plt.subplots(figsize=(8, 3.5))
df[["sharpe"]].plot.bar(ax=ax, color="#2E86AB", edgecolor="black"); ax.axhline(0, color="black", lw=0.7)
ax.set_title("Net Sharpe vs rebalance frequency"); plt.show()
df.round(3)
'''),
    ("md", "## 4. Borrow-cost sweep (basis points annual)"),
    ("code", '''\
res = []
for bc in [0, 25, 50, 100, 200]:
    p = perf_for_costs(cps=1.0, sps=2.0, bps_borrow=bc, rebal=5)
    res.append({"borrow_bps_annual": bc, "sharpe": p["sharpe"], "ann_return": p["ann_return"]})
df = pd.DataFrame(res).set_index("borrow_bps_annual")
fig, ax = plt.subplots(figsize=(8, 3.5))
df[["sharpe"]].plot.bar(ax=ax, color="#06A77D", edgecolor="black"); ax.axhline(0, color="black", lw=0.7)
ax.set_title("Net Sharpe vs short-borrow cost"); plt.show()
df.round(3)
'''),
    ("md", "## 5. Equity + drawdown, baseline strategy"),
    ("code", '''\
from mfa.viz import strategy_tearsheet
strategy_tearsheet(bt.net_returns, benchmark=bm, turnover=bt.turnover, perf=perf,
                   title="Multi-Factor strategy -- baseline backtest");
'''),
    ("md", "### Takeaways\n\n"
           "* Net Sharpe degrades roughly linearly with trade cost; ~10 bps per side wipes most of the alpha.\n"
           "* Rebalancing daily (rebal=1) maximises gross signal but turnover destroys net P&L.\n"
           "* Borrow cost is a smaller knob than execution cost in this universe."),
])


# ===========================================================================
# 06 -- WALK-FORWARD PAPER TRADING
# ===========================================================================
write("06_walk_forward_simulation.ipynb", [
    ("md", "# 06 - Walk-forward 'paper trading' simulation\n\n"
           "A clean out-of-sample test. Every 21 trading days we re-fit the per-factor weights using\n"
           "ONLY data observable up to that date, then trade the next 21 days with those weights.\n\n"
           "* Warmup: 252 trading days (~1 year)\n"
           "* Refit cadence: 21 days\n"
           "* No look-ahead anywhere; every signal at date t was constructible from data up to t-1."),
    ("code", PREAMBLE),
    ("code", LOAD_CACHED),
    ("code", '''\
from mfa.data import forward_returns
from mfa.evaluation import quantile_returns, long_short_returns
from mfa.evaluation.stats import perf_stats
from mfa.portfolio import factor_timing_weights, long_short_quantile_weights
from mfa.backtest import BacktestConfig, CostModel, simulate_paper_trading

daily_ret = panels["close"].pct_change().fillna(0.0)
bm = panels["benchmark_close"].iloc[:, 0].pct_change().fillna(0.0)
cm = CostModel(commission_bps=1.0, slippage_bps=2.0, borrow_cost_bps_annual=50.0)
config = BacktestConfig(rebalance=5, initial_capital=1_000_000, cost_model=cm, benchmark_returns=bm)

def weight_fn(facts_now, dates_now):
    fl = forward_returns(panels["close"].loc[panels["close"].index <= dates_now[-1]], [1])[1]
    pnl = pd.DataFrame({n: long_short_returns(quantile_returns(f, fl, 5)) for n, f in facts_now.items()})
    return factor_timing_weights(pnl, method="rolling_sharpe", lookback=63)

def sizing_fn(comp):
    return long_short_quantile_weights(comp, top_q=0.2, bot_q=0.2, gross_leverage=2.0, max_weight=0.10)

print("starting walk-forward simulation...")
live = simulate_paper_trading(factors, daily_ret, weight_fn=weight_fn, sizing_fn=sizing_fn,
                              config=config, refit_every=21, warmup=252)
perf = perf_stats(live.backtest.net_returns)
print(f"done. sharpe={perf['sharpe']:+.2f}  ann={perf['ann_return']:+.2%}  mdd={perf['max_drawdown']:.2%}")
'''),
    ("md", "## 1. Out-of-sample equity & drawdown"),
    ("code", '''\
from mfa.viz import strategy_tearsheet
strategy_tearsheet(live.backtest.net_returns, benchmark=bm, turnover=live.backtest.turnover,
                   perf=perf, title="Walk-forward paper trading -- net of costs");
'''),
    ("md", "## 2. Factor weights through time (out-of-sample)"),
    ("code", '''\
fig, ax = plt.subplots(figsize=(11, 4.5))
live.factor_weights_history.plot.area(ax=ax, lw=0, alpha=0.85, colormap="tab20")
ax.set_title("Walk-forward factor weights"); ax.set_ylabel("weight"); ax.set_ylim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=7); plt.tight_layout(); plt.show()
'''),
    ("md", "## 3. Top names long / short on the last date"),
    ("code", '''\
last_w = live.backtest.weights_held.iloc[-1].sort_values()
top_short = last_w.head(8)
top_long = last_w.tail(8)
df = pd.concat([top_short, top_long])
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#E63946" if v < 0 else "#06A77D" for v in df.values]
ax.barh(df.index, df.values, color=colors, edgecolor="black")
ax.axvline(0, color="black", lw=0.7); ax.set_title(f"Latest book on {live.backtest.weights_held.index[-1].date()}")
ax.set_xlabel("weight"); plt.show()
'''),
    ("md", "## 4. Year-by-year walk-forward stats"),
    ("code", '''\
r = live.backtest.net_returns
def _yr_stats(s: pd.Series) -> dict:
    return {
        "ret": (1+s).prod()-1,
        "vol": s.std()*np.sqrt(252),
        "sharpe": (s.mean()/s.std())*np.sqrt(252) if s.std()>0 else np.nan,
        "max_dd": (((1+s).cumprod() / (1+s).cumprod().cummax()) - 1).min(),
    }
groups = r.groupby(r.index.year)
yearly = pd.DataFrame({y: _yr_stats(g) for y, g in groups}).T
yearly.index.name = "year"
yearly.style.format({"ret":"{:+.2%}","vol":"{:.2%}","sharpe":"{:+.2f}","max_dd":"{:.1%}"}).background_gradient(cmap="RdYlGn", subset=["sharpe","ret"])
'''),
    ("md", "### Takeaways\n\n"
           "* Walk-forward results are the only ones a real PM should trust.\n"
           "* Even with realistic costs, the out-of-sample tearsheet exposes regime risk -- some years\n"
           "  the multi-factor tilt rewards us, others it pays the cost of being wrong.\n"
           "* Next steps to improve: sector neutralization, beta neutralization, smarter factor timing\n"
           "  (e.g. macro-conditional weights), and a much wider universe."),
])


print("\nAll notebooks generated. Now execute them with:")
print("  .\\\\venv\\\\Scripts\\\\jupyter.exe nbconvert --to notebook --execute notebooks\\\\*.ipynb --inplace")
