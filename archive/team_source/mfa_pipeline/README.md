# Multi-Factor Alpha Pipeline (`mfa_pipeline`)

A self-contained, industry-style multi-factor research framework for US
equities. Pulls free historical daily bars from **Alpaca**, builds a zoo of
classic + creative cross-sectional factors, evaluates them with a full
suite of metrics, combines them with dynamic weights, runs a daily backtest
with realistic costs, and produces publication-quality tearsheets.

The whole folder is **standalone** -- copy `mfa_pipeline/` anywhere on disk,
`pip install -r requirements.txt`, and run.

```
mfa_pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.py                 # one-shot end-to-end runner
├── configs/
│   └── default.yaml                # universe, dates, API keys, knobs
│
├── mfa/                            # the package
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   │
│   ├── data/
│   │   ├── alpaca_loader.py        # cached, incremental REST loader
│   │   ├── universe.py
│   │   └── preprocess.py           # clean/align panels + forward returns
│   │
│   ├── features/
│   │   ├── price.py                # rolling vol/skew/MA/dist-from-high...
│   │   ├── volume.py               # Amihud, turnover proxy, dollar vol
│   │   ├── technical.py            # RSI, MACD, BBands, ATR
│   │   └── crosssectional.py       # rank/zscore/winsorize/neutralize
│   │
│   ├── factors/
│   │   ├── base.py                 # Factor + FactorRegistry
│   │   ├── momentum.py             # 1-12m, 12-1, residual
│   │   ├── reversal.py             # 1d, 5d, gap reversal
│   │   ├── volatility.py           # low-vol, idio-vol
│   │   ├── quality.py              # smoothness, up-day ratio, sharpe
│   │   ├── liquidity.py            # Amihud, dv-momentum, turnover Z
│   │   ├── microstructure.py       # overnight, intraday, range comp.
│   │   ├── novel.py                # pullback-in-uptrend, mood, MAX, ...
│   │   └── compute.py
│   │
│   ├── evaluation/
│   │   ├── ic.py                   # IC, IR, t-stat, decay
│   │   ├── quantiles.py            # 5-quantile portfolios + turnover
│   │   ├── stats.py                # Sharpe/Sortino/Calmar/VaR/CVaR/...
│   │   ├── attribution.py          # cross-factor regression + corr
│   │   └── reports.py              # one-call per-factor report
│   │
│   ├── portfolio/
│   │   ├── combiner.py             # standardize + weighted combine
│   │   ├── timing.py               # rolling-Sharpe / IC factor timing
│   │   ├── risk.py                 # Sample / Ledoit-Wolf / EWMA cov
│   │   └── optimizer.py            # quantile L/S, mean-variance, RP
│   │
│   ├── backtest/
│   │   ├── engine.py               # daily, drift-aware, cost-aware
│   │   ├── execution.py            # commissions + slippage + borrow
│   │   └── live_sim.py             # walk-forward paper trading
│   │
│   └── viz/
│       ├── style.py
│       ├── factor_viz.py
│       ├── perf_viz.py
│       └── tearsheet.py            # multi-panel tearsheets
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_factors.py
│   ├── 03_evaluate_factors.py
│   ├── 04_combine_and_backtest.py
│   └── 05_make_tearsheet.py
│
├── notebooks/
│   └── 00_end_to_end_demo.ipynb    # full walkthrough with charts
│
└── data/
    ├── raw/                        # Alpaca parquet cache
    ├── processed/                  # aligned panels + factor parquets
    └── results/                    # tearsheets, csvs, equity curves
```

---

## Quick start

### A. Virtual environment (recommended)

```powershell
# from the mfa_pipeline/ folder
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
# (optional) register the venv as a Jupyter kernel
.\venv\Scripts\python.exe -m ipykernel install --user --name mfa-venv --display-name "Python (mfa_pipeline venv)"
```

### B. Set Alpaca paper-trading keys (free)

Edit `configs/default.yaml` -> `alpaca: api_key / secret_key`, OR export:
```powershell
$env:ALPACA_API_KEY  = "PK..."
$env:ALPACA_SECRET_KEY = "..."
```

### C. Run the pipeline

```powershell
.\venv\Scripts\python.exe run_pipeline.py
# … or step by step:
.\venv\Scripts\python.exe scripts\01_download_data.py
.\venv\Scripts\python.exe scripts\02_build_factors.py
.\venv\Scripts\python.exe scripts\03_evaluate_factors.py --horizon 5
.\venv\Scripts\python.exe scripts\04_combine_and_backtest.py
.\venv\Scripts\python.exe scripts\05_make_tearsheet.py
```

### D. Research notebooks

After steps C-1 and C-2 have populated the cache, every notebook runs in
seconds. Open them in Cursor / Jupyter and pick the `Python (mfa_pipeline venv)`
kernel.

```
notebooks/
├── 00_end_to_end_demo.ipynb           # full pipeline walkthrough
├── 01_data_eda.ipynb                  # data quality + return distributions + dispersion
├── 02_factor_research_classic.ipynb   # IC scoreboard for momentum/reversal/vol/...
├── 03_factor_research_novel.ipynb     # the 8 brain-storm factors, deep dive each
├── 04_combination_and_timing.ipynb    # equal vs IC vs rolling-Sharpe weighting
├── 05_backtest_costs.ipynb            # cost / rebalance / borrow sensitivity
├── 06_walk_forward_simulation.ipynb   # out-of-sample paper trading
└── 07_iteration_v2.ipynb              # honest train/test iteration: -1.36 → +0.11 OOS Sharpe
```

To re-execute every notebook in a CI-friendly way:
```powershell
foreach ($nb in (ls notebooks\*.ipynb)) {
    .\venv\Scripts\jupyter.exe nbconvert --to notebook --execute $nb.FullName --inplace --ExecutePreprocessor.timeout=1800
}
```

---

## Pipeline overview (what each layer does and why)

### 1. Data acquisition (`mfa.data.alpaca_loader`)
Pure REST against `data.alpaca.markets/v2/stocks/{sym}/bars`. Per-symbol
parquet cache; reruns only fetch the missing tail. Built-in retries with
exponential back-off for 429s and 5xxs.

### 2. Preprocessing (`mfa.data.preprocess`)
* drop names with insufficient history,
* short forward-fill for tiny gaps (no extrapolation),
* align all panels to a common date / symbol grid,
* split off the benchmark series,
* compute non-overlapping forward returns at multiple horizons.

### 3. Features (`mfa.features.*`)
Vectorized building blocks: rolling stats, RSI/MACD/BBands/ATR, dollar
volume, Amihud illiquidity, plus per-row cross-sectional rank/zscore/winsor
helpers used to standardize signals before combining.

### 4. Factor zoo (`mfa.factors.*`)
~18 factors across 7 families with a tiny registry so adding a new one is
just a decorator:

```python
from mfa.factors.base import REGISTRY

@REGISTRY.register(name="my_factor", family="novel",
                   description="...", default_params={"window": 63})
def my_factor(panels, window=63):
    c = panels["close"]
    ...
    return signal_panel
```

Highlights of the *creative* factors in `factors/novel.py`:
* `pullback_in_uptrend` -- shallow pullbacks inside healthy uptrends
* `acceleration_21_63` -- 2nd derivative of price (short - long mean ret)
* `vol_price_confirmation_21` -- corr(daily ret, daily volume change)
* `negative_skew_63` -- inverse-skew (lottery anomaly)
* `mood_63` -- distribution-shape composite using skew and kurt
* `max_return_inverse_21` -- Bali-Cakici-Whitelaw "MAX" anomaly
* `distance_from_extremes_63` -- mean-reversion inside a channel
* `weekly_seasonal_friday` -- end-of-week flow effect

### 5. Evaluation (`mfa.evaluation.*`)
Comprehensive: per-factor cross-sectional Spearman IC + IR + t-stat + hit
rate; IC decay across horizons; 5-quantile cumulative returns; long-short
PnL with full risk metrics (Sharpe, Sortino, Calmar, max DD, max DD
duration, VaR/CVaR, skew/kurt); top-quantile turnover; cross-factor
correlation matrix; OLS attribution of strategy returns to factor returns.

### 6. Combination + timing (`mfa.portfolio.*`)
* **standardize** every factor cross-sectionally (winsorize + zscore),
* **combine** with static or *time-varying* per-factor weights,
* **time** factors via rolling Sharpe / rolling IC / rolling cum-return / **`ic_signed`**
  (sign-aware: anti-IC factors contribute with flipped sign instead of being clipped to zero),
* **build** target weights via long-short quantile (default), mean-variance (SLSQP) or risk-parity,
* **risk** model: Sample / Ledoit-Wolf / EWMA covariance,
* **point-in-time guard**: every weight panel is `.shift(1)`-ed unless the input is
  flagged `pnl_already_pit=True`, so weights at date t cannot peek at t's PnL.

### 6b. Anti-overfitting toolkit (`mfa.portfolio.{selection,sizing,neutralize}`)
* `split_dates(idx, train_end[, val_end])` -- enforce a frozen train/(val)/test split.
* `factor_train_summary(...)` -- IC summary computed **only on train**, with a Bonferroni-style
  family-wise threshold for `|t|`.
* `select_factors_train(...)` -- returns the surviving factor list to be used downstream.
* `target_vol_sizing(weights, returns, target_ann_vol=0.10, lookback=63)` -- rescale gross
  leverage so realised vol tracks the target; both the rolling vol *and* the scale factor
  are `.shift(1)`-ed.
* `market_beta_neutralize_signal(signal, returns, bm_ret, beta_window=252)` -- residualise
  the cross-sectional signal against rolling market beta.
* `apply_turnover_penalty(weights, shrink)` -- convex blend with yesterday's weights.

### 7. Backtester (`mfa.backtest.*`)
Daily, vectorized, drift-aware. Costs charged on every rebalance:
`commission_bps + slippage_bps` on traded notional + `borrow_cost_bps_annual`
prorated on average short notional. Returns gross + net curves, weights
held, turnover, costs, plus a walk-forward "paper-trading" simulator
(`simulate_paper_trading`) that re-fits factor weights every N days using
*only* point-in-time information.

### 8. Visualization (`mfa.viz.*`)
Cohesive matplotlib style (`apply_style`). Plots for IC series & decay,
quantile cumulative returns, factor correlation heatmap, equity, drawdown,
rolling Sharpe, monthly heatmap, annual returns, turnover. Two ready-made
tearsheets: `factor_tearsheet` (per-factor) and `strategy_tearsheet`
(per-strategy with all panels in one figure).

---

## Configuration

Everything tunable lives in `configs/default.yaml`. Key knobs:

| section       | knob                          | what it does |
|---------------|-------------------------------|--------------|
| `universe`    | `symbols`, `benchmark`        | which stocks |
| `period`      | `start`, `end`, `timeframe`   | date span    |
| `evaluation`  | `n_quantiles`, `ic_method`    | factor analytics |
| `combine`     | `method` (`equal`/`ic_weight`/`ic_ir_weight`/`ridge`) | how to merge factors |
| `portfolio`   | `top_quantile`, `gross_leverage`, `rebalance`, `max_weight` | sizing |
| `risk`        | `cov_method`                  | covariance estimator |
| `timing`      | `signal`, `lookback`          | factor timing logic |
| `backtest`    | `commission_bps`, `slippage_bps`, `borrow_cost_bps_annual` | cost model |

---

## Notes on data quality

* Alpaca free tier uses the **IEX feed**: a single venue (~3% of US tape).
  Volumes are tiny vs. SIP; absolute liquidity-based factors may be noisier
  than they would be on consolidated tape.  Cross-sectional **ranks** of
  those factors are still informative -- the relative ordering is preserved.
* Bars are split-adjusted (`adjustment=split`).  Dividends are not adjusted
  for in this download; for total-return strategies you'll want to layer in
  a dividend feed.
* Timestamps are stored in UTC and normalized to midnight for daily bars,
  so the date column lines up cleanly across symbols.

---

## License & disclaimer

This is research code. No part of it is investment advice. The author makes
no claim about live profitability of any strategy or factor included here.
Backtests in academic settings are routinely overstated; use the
walk-forward simulator and apply your own out-of-sample sanity checks.
