# QTC: Intraday Residual Reversal and Validation

A research archive for an intraday residual-reversal strategy in AAPL, AMD, MSFT and NVDA, using XLK as the primary sector proxy. The original team source is attributed to https://github.com/rentiankai2003/imperial-qtc-trading-strategy-research . Archiving it here does not imply sole authorship or grant a new license.

## Final conclusion
Do not deploy the retained candidate with real capital. The team validation report withdrew the earlier deployment recommendation after multi-regime tests and continuous-panel checks. Earlier favorable Sharpe ratios are intermediate findings, not the final expected performance.

## Method
Estimate trailing 390-bar beta, remove the sector component, accumulate residuals over 30 bars, standardize against trailing history, and use sparse entries/exits instead of minute-by-minute portfolio rebalancing. The archived beta implementation uses a zero-intercept estimator; it should not be described as an intercept-inclusive regression without checking the code.

Warm-up history matters: a regime filter calibrated only within a short evaluation window can suppress trades and create misleading performance. The source report documents corrected 2024 Sharpe of -0.73 after adequate history, versus a prior +2.21. These figures are quoted from the team report, not reproduced afresh in this packaging run.

## Run
`python src/reproduce.py` checks a known-beta synthetic series, lagged estimation and full-window z-score warm-up. Full historical PnL is not reproduced because the original minute panel is absent. Archived research and validation code is under `archive/team_source/`; data downloads and trading-platform code are not automatically executed.
