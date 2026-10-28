# Imperial QTC Intraday Trading Strategy Research

This repository contains the research, implementation, and validation of an
intraday residual-reversal strategy developed for Queen's Tower Capital (QTC).
The project studies whether temporary stock-specific moves in large-cap US
technology equities revert after removing their common exposure to XLK.

> **Research status:** do not deploy the current strategy with real capital.
> Selected 2025-2026 samples were positive, but continuous-panel and
> multi-regime stress tests found negative performance in 2022, 2023, and 2024.
> The repository preserves both the promising research path and the evidence
> that overturned the original deployment recommendation.

## Strategy Summary

The target universe is AAPL, AMD, MSFT, and NVDA. XLK is the primary sector
proxy, with QQQ as a fallback.

For each stock, the strategy:

1. Estimates a rolling 390-minute beta to XLK.
2. Removes the common sector return to obtain a stock-specific residual.
3. Accumulates residual returns over a 30-minute signal horizon.
4. Standardizes the signal against its trailing 390-minute distribution.
5. Takes contrarian positions when the residual z-score is extreme.
6. Uses a sparse state machine to reduce turnover and transaction costs.
7. Applies residual-dispersion regime logic and execution guardrails.

The platform-facing candidate in `strategy.py` uses a 2.75 entry threshold, a
120-minute maximum holding period, and long-only execution. These defaults are
retained for integration testing and are not a live-trading recommendation.

## Research Trajectory

| Stage | Main result | Interpretation |
| --- | ---: | --- |
| Naive momentum baseline | OOS Sharpe -3.95 | The initial trend hypothesis failed. |
| Continuous residual reversal | Net Sharpe -7.45 | Gross edge was consumed by turnover. |
| Sparse state machine | Large turnover reduction | Execution design made the signal testable. |
| Narrow Dec 2025-Mar 2026 window | Sharpe +2.72 | Promising but window-dependent. |
| Phase-12 original test window | Sharpe +2.17, MaxDD -4.8% | Later found not to generalize. |
| Corrected 2024 OOS | Sharpe -0.73 | Proper pre-history removed a warm-up artifact. |
| 2022 technology bear market | Sharpe -2.06 | Negative outside the favorable regime. |
| May-Dec 2023 | Sharpe -2.93 | Further evidence of regime dependence. |

See `RESEARCH.md` for the full research narrative and
`VALIDATION_REPORT.md` for the final stress-test decision.

## Repository Layout

- `strategy.py`: QTC-compatible live strategy shell and self-test.
- `src/`: research backtester and residual-reversal implementation.
- `notebooks/`: EDA, ablations, strategy pivot, honest revisit, and scorecard.
- `validation/`: OOS, continuous-panel, stress-regime, and robustness scripts.
- `data_qtc/validation/`: compact JSON and CSV validation summaries.
- `results/`: compact research tables; large generated arrays are excluded.
- `mfa_pipeline/`: standalone multi-factor alpha research pipeline.
- `fetch_data.py`: market-data acquisition and panel-building utilities.
- `RESEARCH.md`: detailed research log and methodology.
- `VALIDATION_REPORT.md`: final validation report and deployment retraction.

## Installation

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The multi-factor pipeline has its own dependency file:

```powershell
pip install -r mfa_pipeline\requirements.txt
```

## Data

Raw minute bars, virtual environments, and generated Parquet panels are not
stored in Git because they are large and reproducible. Use `fetch_data.py` and
the scripts under `validation/` to rebuild the required datasets. Alpaca data
access requires credentials supplied through the local environment; never
commit API keys to the repository.

Compact validation summaries are retained so that the reported conclusions can
be audited without downloading the full market-data panel.

## Validation Principles

The final validation uses continuous historical panels so rolling regime
statistics have genuine pre-period history. Evaluation includes transaction
cost stress, walk-forward testing, block bootstrap, parameter grids,
drop-best-period checks, and independent market-regime tests.

The central lesson is that a strong Sharpe ratio in one favorable window is not
sufficient evidence of deployable alpha. A future candidate should remain
positive across independent regimes, survive realistic costs, and complete a
fully frozen forward paper-trading period before any live-capital discussion.

## Disclaimer

This repository is for research and educational purposes only. It does not
constitute investment advice, and the current strategy is not approved for live
deployment.
