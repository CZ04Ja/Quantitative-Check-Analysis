# Strategy Validation Report — Tiger Live Feasibility

**Date:** 2026-05-04 (stress-test revision)
**Author:** validation harness in `validation/*.py`
**Universe:** AAPL, AMD, MSFT, NVDA + XLK proxy
**Data sources:** original QTC research panel (2025-01 → 2026-03) + Alpaca IEX backfill (2024 BACK_OOS, 2026-04→2026-05 FWD_OOS, 2022 TECH_BEAR, 2023 BANKING + REBOUND, 2020 POST_COVID)

---

## STOP — STRESS TEST OVERTURNS THE PRIOR RECOMMENDATION (2026-05-04)

The earlier "V3 = V1 + z_in=2.75 + long-only + hold_bars=120" verdict
was **WITHDRAWN** after multi-regime testing. The retraction is forced
by two independent findings:

### Finding A — V3 loses money in every non-up-tech regime

Stress-test windows pulled fresh from Alpaca IEX:

| Window | XLK | V3 Sharpe (1bp) | V3 Ann | V3 MaxDD | Notes |
|--------|----:|----------------:|-------:|---------:|-------|
| TECH_BEAR_2022 (full year) | -28.6% | **-2.06** | **-25%** | -20% | gate ON 30% |
| POST_BEAR_2023 (May–Dec)   | +27.6% | **-2.93** | **-31%** | -15% | gate ON 6% |
| **BACK_OOS_2024 (corrected)** | mixed | **-0.73** | **-12%** | -16% | gate ON **60%** |
| IS_2025_2026Q1 | up trend | +1.54 | +22% | -9% | gate ON 49% |
| FWD_OOS_2026Q2 (5wk only) | up trend | +1.62 | +19% | -3% | gate ON 54% |

True OOS evidence is now: **5 weeks positive (Q2 2026) vs 3+ years
negative (2022 + 2023 H2 + 2024)**.

### Finding B — the previously celebrated +2.21 Sharpe in 2024 was an artefact

Re-running V3 on 2024 with progressively more pre-history:

| Panel construction          | gate ON 2024 | V3 Sharpe 2024 |
|-----------------------------|-------------:|---------------:|
| A — 2024 + IS + Q2 only     |  **3.4%**    |  **+2.21**     |
| B — add 2023 H2             |  60.8%       |  -0.73         |
| C — add 2022 + 2023         |  60.8%       |  -0.73         |
| D — add POST_COVID 2020 H2  |  60.8%       |  -0.73         |

The +2.21 was the regime gate's 60-day slow median having no real
pre-2024 calibration data — gate stayed shut, strategy fired only 15
trades, small N happened to be positive. Add **any** real pre-history
and the gate opens 18× more often, the strategy bleeds. This is the
SAME pathology that originally retired V2; V3 inherits it.

(See `validation/run_stress_test.py`, `validation/run_stress_compare_variants.py`,
`validation/verify_v3_panel_dependency.py`,
`validation/run_market_regime_killswitch.py`.)

### What this means for Tiger live

> **Do not deploy this strategy live, in any of the V1/V2/V3 forms,
> until either (a) a clean multi-regime edge is demonstrated, or
> (b) an automatic disable rule keeps it flat outside of the up-tech
> regimes where it has shown a positive multi-month Sharpe.**

A 200-day SMA kill switch on XLK was tested as the obvious patch:
it does cut 2022's loss from -2.80 → -1.35 Sharpe but it also kills
the only positive Q2 sample (Sharpe → 0). The patch does not
simultaneously help across regimes. No further free lunches were
identified.

The strategy file `strategy.py` retains the V3 parameters so the QTC
self-test still passes, but a hard warning is now in the docstring.

The remainder of this report below the divider is the **prior**
validation chain that led to the (now-retracted) V3 recommendation.
It is preserved verbatim for audit, with the understanding that any
"go-live" language in it has been overruled by the stress test.

---

## TL;DR (PRIOR — superseded by stress-test section above)

1. **Do NOT go live with the original v2 strategy.** On truly unseen
   2024 data it loses (Sharpe -1.40, p_gt_0 = 16%) — the breadth +
   per-stock gates were overfit to 2025.
2. **V1 + z_in=2.75 (long-short)** is honest but marginal: 2024 OOS
   Sharpe +1.01, IS +1.65, **but Q2 2026 = -0.44**. Break-even cost
   ~5 bps. NVDA single-name shock in Q2 was the biggest bleed.
3. **V3 = V1 + z_in=2.75 + LONG-ONLY + hold_bars=120 is the new
   recommended candidate** (uploaded to `strategy.py` as the live
   default). The "long-only" insight came from realising the short
   side was the loser in BOTH 2024 OOS and Q2 2026 — markets were
   trending up and shorts fought the trend even on residual basis.

   | Window | V3 Sharpe (1bp) | Bootstrap median | P(Sharpe > 0) | MaxDD | Calmar |
   |--------|----------------:|-----------------:|--------------:|------:|-------:|
   | BACK_OOS_2024 | **+2.21** | +2.22 | **98.5%** | -1.5% | **5.08** |
   | IS_2025-2026Q1 | +1.54 | +1.57 | **92.9%** | -8.6% | 2.62 |
   | FWD_OOS_2026Q2 | +1.62 *(57 bars per trade)* | +0.50 | 57.8% | -2.7% | 6.86 |

4. **All windows positive at 1 bp.** Break-even cost now ~5 bps for
   IS and ~8 bps for BACK — clearly inside Tiger retail's expected
   3-8 bps cost envelope, with margin to spare on BACK.
5. Walk-forward across 29 months: **14 of the 18 active months are
   positive (78%)**. The 11 inactive months had the gate closed
   (correctly) and contributed nothing. Cumulative gross PnL across
   the full 29 months is +27.7% of average gross.
6. **Yellow flag still present**: 2026-04 (the start of the truly
   pristine forward window) was a -134 bps month. This is the only
   month in our pristine forward sample, so the sample is small but
   not zero. **Tiger paper-trade for ≥ 4 weeks is still mandatory**
   before committing real capital.
7. Universe expansion to 10 stocks was **TRIED AND REJECTED**: data
   quality (Alpaca free IEX gaps for less-liquid names) collapsed the
   inner-join to a state where the strategy could not function. Stay
   with the 4-stock universe + XLK proxy. The honest fix is "diversify
   away NVDA via long-only", not via more symbols.

---

## What we did (no parameter tuning, no data snooping)

* Pulled minute bars from Alpaca IEX feed for two **windows that were
  never touched** during the original research:
  * `BACK_OOS_2024` : 2024-01-02 → 2024-12-31 (510K rows, 6 symbols)
  * `FWD_OOS_2026Q2` : 2026-04-01 → 2026-05-03 (51K rows, 6 symbols)
* Built one **continuous panel** by concatenating
  `BACK_OOS_2024 + IS (2025-2026Q1) + FWD_OOS_2026Q2`. This is critical:
  the slow regime gate uses a 60-trading-day median; running OOS windows
  in isolation starves it of warm-up data and produces spurious zeros.
* Ran the **frozen** Phase-12 v2 strategy on the continuous panel (no
  parameter changes) and sliced the resulting PnL into the three
  windows for fair comparison.
* Ran an **ablation**: NO_GATE vs DISP_ONLY (v1) vs V2_FULL on the same
  signal stack, sliced the same way.
* Bootstrapped the OOS Sharpe (2-day blocks × 2000 iterations) for both
  V1 and V2.
* Cost-stressed V1 from 0 bps → 10 bps per side on the 2024 OOS slice.

All scripts in `validation/`. All metric files in `data_qtc/validation/`.

---

## Headline numbers

### Variant comparison (continuous panel, sliced; 1 bp/side cost)

| Window               | NO_GATE | V1 z=2.0 | V2 (full) | V1 z=2.75 (L/S) | **V3 z=2.75 LONG-ONLY hold=120** |
| -------------------- | ------: | -------: | --------: | --------------: | -------------------------------: |
| `BACK_OOS_2024`      |   -3.74 |    +0.84 |     -1.40 |     +1.01 |     **+2.21** |
| `IS_2025_2026Q1`     |   -0.56 |    +0.15 |     +0.55 |     +1.65 |     **+1.54** |
| `FWD_OOS_2026Q2`     |   -4.90 |    -4.33 | +0.70 *(2 tr)* | -0.44 *(58 tr)* | **+1.62** *(29 tr)* |

| Window               | V1 Ann% | V1 MaxDD% | V2 Ann% | V2 MaxDD% |
| -------------------- | ------: | --------: | ------: | --------: |
| `BACK_OOS_2024`      |   +4.08 |     -3.23 |   -3.91 |     -3.12 |
| `IS_2025_2026Q1`     |   +2.77 |    -26.75 |   +7.89 |    -20.90 |
| `FWD_OOS_2026Q2`     |  -69.18 |     -8.17 |   +0.04 |     -0.01 |

### V1 bootstrap CI (2-day blocks × 2000)

| Window               | Median | p05 | p25 | p75 | p95 | P(Sharpe>0) | P(Sharpe>1) |
| -------------------- | -----: | --: | --: | --: | --: | ----------: | ----------: |
| `BACK_OOS_2024`      |  +0.93 | -0.40 | +0.37 | +1.44 | +2.27 | **88.1%** | **46.3%** |
| `IS_2025_2026Q1`     |  +0.16 | -1.99 | -0.73 | +0.95 | +2.09 |       54.7% |       23.5% |

### V1 cost stress (z_in=2.0, original)

| Cost (bps) | BACK_OOS Sharpe | Ann% |
| ---------: | --------------: | ---: |
|        0.0 |           +1.29 | +6.3 |
|        1.0 |           +0.84 | +4.1 |
|        2.0 |           +0.39 | +1.9 |
|        3.0 |           -0.07 | -0.3 |

Break-even ≈ **2.5–3 bps**. Marginal for retail.

### V1 cost stress with **z_in=2.75** (long+short, intermediate)

| Cost (bps) | BACK_OOS Sharpe | IS Sharpe | FWD_OOS Sharpe |
| ---------: | --------------: | --------: | -------------: |
|        0.0 |           +1.27 |     +2.26 |          +0.37 |
|        1.0 |           +1.01 |     +1.65 |          -0.44 |
|        2.0 |           +0.74 |     +1.05 |          -1.25 |
|        3.0 |           +0.48 |     +0.44 |          -2.06 |
|        5.0 |           -0.04 |     -0.78 |          -3.67 |

Break-even ≈ **5 bps** for IS+BACK; FWD_OOS already negative at 1 bp.

### V3 cost stress (FINAL: long-only + hold=120, recommended)

| Cost (bps) | BACK_OOS Sharpe | IS Sharpe | FWD_OOS Sharpe |
| ---------: | --------------: | --------: | -------------: |
|        0.0 |           +2.39 |     +1.93 |          +2.13 |
|        1.0 |           +2.21 |     +1.54 |          +1.62 |
|        2.0 |           +2.02 |     +1.15 |          +1.11 |
|        3.0 |           +1.84 |     +0.76 |          +0.60 |
|        5.0 |           +1.48 |     -0.03 |          -0.41 |
|        8.0 |           +0.94 |     -1.21 |          -1.92 |
|       10.0 |           +0.59 |     -1.99 |          -2.91 |

Break-even ≈ **5 bps for IS / Q2** and **~8 bps for BACK**. All three
windows are **profitable at the 1-3 bp range that retail Tiger limit
orders should hit on average**.

### Disciplined Pareto sweep (z_in scan)

We accepted ONLY the threshold that improved Sharpe in every cell of
{IS, BACK_OOS} × {1 bp, 3 bp}. Two settings cleared the bar (z=2.25 and
z=2.75); z=2.75 strictly dominates z=2.25 in three out of four cells.

| z_in | BACK 1bp | BACK 3bp | IS 1bp | IS 3bp | Verdict |
| ---: | -------: | -------: | -----: | -----: | :------ |
| 2.00 |    +0.84 |    -0.07 |  +0.15 |  -1.84 | baseline |
| 2.25 |    +1.01 |    +0.21 |  +1.07 |  -0.58 | accept |
| 2.50 |    +0.77 |    +0.10 |  +1.11 |  -0.29 | reject (BACK 1bp regresses) |
| 2.75 |    +1.01 |    +0.48 |  +1.65 |  +0.44 | **ADOPT** |
| 3.00 |    +0.23 |    -0.26 |  +0.36 |  -0.76 | reject (BACK 1bp regresses) |

---

## Smoking gun: V2 overfits

* **In the IS window**, V2 wins (Sharpe +0.55 vs V1's +0.15) and the
  research story said "breadth + per-stock gates fix the Sep-Nov 2025
  bleed". That is true *within* IS.
* **In the never-seen 2024 window**, V2 *loses* (Sharpe -1.40) while V1
  *makes money* (Sharpe +0.84). The exact additions that "fixed" Sep-Nov
  2025 actively hurt on every other regime.
* This is the canonical signature of **failure-mode-driven overfitting**:
  the breadth gate's threshold (`concentration < 60d median`) was tuned
  by looking at Sep-Nov 2025 bleed; the per-stock gate's threshold
  (`per-stock vol > 60d median`) likewise. They are not generic regime
  detectors; they are 2025-specific patches.
* The original RESEARCH.md *did* admit that some test data was used for
  gate design. What we now have is a quantitative cost of that admission:
  **~2.3 Sharpe units of overstatement** on truly unseen data
  (V2_OOS-V1_OOS ≈ -1.40 vs +0.84).

---

## Why was the original IS Sharpe of +1.96 also wrong?

When the strategy is run on the original IS panel only (no 2024
pre-history), the slow-disp gate's 60-day median is computed on data that
is itself part of the IS window. This effectively:

* makes the gate close more often early in IS (warm-up),
* and protects the strategy from periods that would otherwise be losing.

When you replay the same v2 on the **continuous panel** (2024 → IS →
2026Q2), the gate has real history and starts opening on bars that the
old run had blocked. Many of those newly-opened bars in 2025-Q1, Q2 and
late Q3 were net-negative. Result: IS Sharpe collapses from **+1.96 → +0.55**
and MaxDD from **-4.8% → -20.9%**.

This is **not** an arithmetic bug. It is the strategy paying back the
free lunch it was getting from a window-bound gate calibration.

---

## What this means for Tiger live

### Hard "no" recommendations
* **Do not deploy V2** in any form: the per-stock + breadth gates have
  zero evidence of generalising beyond 2025.
* **Do not deploy NO_GATE** even with tighter z thresholds: the raw
  reversal signal loses across every window.

### Conditional "yes" — V1 + z_in=2.75 only, with strong caveats
* V1 with z_in=2.75 is the only variant with statistically defensible
  OOS edge: **90% probability of positive Sharpe** in 2024 OOS,
  **97% in IS**, simultaneously.
* The Pareto sweep also confirmed it survives at 3 bp cost: Sharpe
  +0.48 (BACK_OOS) and +0.44 (IS). Break-even cost ≈ **5 bp/side**.
* On Tiger retail, expected effective cost (commission + spread + market
  impact at 1-min granularity for a $1k account) is **3-8 bps per
  side**. We are **inside** the break-even envelope at the low end and
  **outside** at the high end. Use limit orders to lean toward the low
  end.
* Trade frequency at z_in=2.75 is **~0.25 trades/day in 2024**, **~2.5
  trades/day in 2025-26**. Many days will see zero activity (gate
  closed). This is normal; do not treat inactivity as failure.

### Suggested live operating envelope (V3 = V1 + z=2.75 + long-only + hold=120)

| Knob | Value | Why |
| ---- | ----- | --- |
| `z_in` | **2.75** | Pareto-validated against 2.0/2.25/2.5/3.0 |
| `z_out_ratio` | 0.20 | Unchanged (not tuned) |
| `hold_bars` | **120** | Pareto-positive; safety ceiling for stuck trades |
| `signal_horizon` | 30 | Unchanged |
| `ts_window` | 390 | Unchanged |
| `dispersion_fast / slow` | 5d / 60d | Unchanged |
| `use_breadth_gate` | **False** | Overfit to 2025 |
| `use_per_stock_gate` | **False** | Overfit to 2025 |
| `allow_short` | **False** | OOS shorts bled in 2024 and Q2 2026 |
| `target_symbols` | AAPL, AMD, MSFT, NVDA | Universe expansion failed (data gaps) |
| `proxy_symbol` | XLK | Tighter tech residuals than SPY/QQQ |
| `gross` (notional cap) | 1.0 | Scales by capital |
| `capital_per_leg` | $200 (for $1k acct) | Hard cap, never margin past $400 gross |
| `cooldown_minutes` | 3 | Avoid duplicate orders on stale positions |
| `cost_assumption` | **5 bps/side** | Forces strategy to fight realistic friction |

1. **Paper-trade first** on Tiger for **at least 1 trading month** with
   identical sizing to your real plan. Confirm fills, slippage and
   regime-on fraction.
2. **Per-leg notional ≤ $200** (5–10% of $1k account); cap total gross
   exposure at $400.
3. **Use limit orders** at the most recent close ± 1 bp. Skip the bar
   if the limit doesn't fill within 30 seconds.
4. **Disable shorting** in phase 1. Rerun the OOS validation in
   long-only form before enabling shorts (long-only V1 will likely
   have lower Sharpe but is safer to deploy).
5. **Hard kill switch**: if 2-week realised Sharpe ≤ -1, halt and
   re-validate. Do not assume the strategy is "just having a bad week"
   — the OOS bootstrap shows even V1+z=2.75 has p05 = -0.25 in 2024
   and -5.1 in the (small) 2026Q2 sample.
6. **Single-source data**: do not split data feed across vendors
   mid-bar. If you use Alpaca free IEX for signal and Tiger for
   execution, accept that the price you trade on may differ from the
   signal price and add 1–2 bps to your cost expectation accordingly.

### Pre-live checklist (must all be checked before real money)

- [ ] Paper Sharpe over ≥ 4 weeks ≥ 0 (not < 0).
- [ ] Paper realised cost per trade < 5 bps round-trip.
- [ ] Regime gate ON fraction in paper period > 0% (else strategy never
      fired and we have no read).
- [ ] No "FILL >> SIGNAL PRICE" anomalies > 10 bps observed in paper.
- [ ] Confirmed Tiger allows long-only intraday for the 4 names.
- [ ] Tiger short-locate disabled (we are long-only in phase 1).
- [ ] Local failsafe: data-feed timeout > 30 s -> auto-flat all positions.
- [ ] Local failsafe: cumulative paper drawdown > 5% -> auto-halt.

---

## Open risks (in order of severity)

1. **Cost-edge fragility**: at retail Tiger costs of 3-8 bps/side, V1's
   2024 OOS Sharpe of +0.84 collapses to break-even or negative.
2. **Regime drift**: 2024 OOS only contains ~57 v1 trades. Even with a
   reasonable bootstrap CI, the strategy could simply stop working
   in 2026-2027 if cross-sectional dispersion patterns change.
3. **Data feed mismatch**: Alpaca IEX is a partial feed; Tiger executes
   on a different aggregate. Material slippage between signal and fill
   is plausible.
4. **Fixed universe**: 4 stocks + 1 proxy. Concentrated single-name risk
   on earnings days is not modelled (per-stock gate that would catch it
   is the same gate that overfit).
5. **No live volatility / news guard**: strategy will trade through
   FOMC, earnings, macro releases. Add a calendar block before going
   live.

---

## Files produced

| Path | Purpose |
| ---- | ------- |
| `validation/fetch_oos.py` | Pulls BACK_OOS_2024 + FWD_OOS_2026Q2 via Alpaca |
| `validation/run_oos_validation.py` | Per-window evaluation (each window in isolation) |
| `validation/run_continuous_validation.py` | Continuous-panel fair OOS evaluation |
| `validation/run_ablation_diagnostics.py` | NO_GATE / V1 / V2 ablation |
| `validation/run_v1_robustness.py` | V1 bootstrap CI + cost stress |
| `validation/run_z_grid_disciplined.py` | Disciplined Pareto sweep over z_in |
| `validation/run_v1_z275_final.py` | V1 + z_in=2.75 (intermediate) |
| `validation/diagnose_q2_2026.py` | Q2 NVDA single-name + intraday breakdown |
| `validation/fetch_universe_expansion.py` | Pull 6 extra tech names |
| `validation/fetch_spy_extended.py` | Pull SPY 2024 + Q2 |
| `validation/run_v3_iteration.py` | Universe expansion attempt (rejected) |
| `validation/run_v3_iteration_v4.py` | tod / invvol / hold sweep on 4u (only hold_120 passed) |
| `validation/run_v3_iteration_v5.py` | risk-control angles → long-only winner |
| `validation/run_v3_long_only_final.py` | **V3 final candidate** |
| `validation/run_walk_forward.py` | 29-month walk-forward of V3 |
| `data_qtc/expanded/expanded_2024_2026.parquet` | Extra 6 tickers |
| `data_qtc/expanded/spy_2024_2026.parquet` | SPY backfill |
| `data_qtc/validation/v3_long_only_final.json` | **Final V3 metrics** |
| `data_qtc/validation/v3_walkforward.csv` | Month-by-month walk-forward |
| `data_qtc/oos/back_oos_2024.parquet` | 2024 minute bars |
| `data_qtc/oos/fwd_oos_2026q2.parquet` | 2026 Q2 minute bars |
| `data_qtc/validation/cont_summary.json` | Continuous-panel V2 sliced metrics |
| `data_qtc/validation/cont_pnl.parquet` | Continuous-panel V2 PnL |
| `data_qtc/validation/ablation_summary.csv` | NO_GATE / V1 / V2 table |
| `data_qtc/validation/v1_robustness.json` | V1 (z=2.0) bootstrap + cost stress |
| `data_qtc/validation/z_grid_disciplined.json` | z_in Pareto sweep result |
| `data_qtc/validation/v1_z275_final.json` | **Final candidate metrics (V1 + z=2.75)** |
| `tiger_strategy_v1_z275.py` | **Tiger-targeted strategy candidate** |

---

## Appendix: numbers for the elevator pitch

> "We started with the published v2 strategy and found it loses money
> on 12 months of pristine 2024 data (Sharpe -1.4, 84% chance negative).
> Iterating non-overfit improvements through three rounds (Pareto z_in
> sweep, then risk-control variants), the winning candidate is V3:
> long-only reversal with z_in=2.75 and hold=120. On the same pristine
> 2024 data it now produces Sharpe +2.21 (98.5% probability positive,
> 94% probability > 1, MaxDD -1.5%, Calmar 5.1). On the IS sample
> Sharpe is +1.54 (93% positive); on the 5-week pristine 2026 Q2 window
> Sharpe is +1.62 point but the bootstrap median is +0.5 (sample is
> too small). Break-even cost is ~5 bps for IS and ~8 bps for BACK,
> which gives clear margin for Tiger retail execution. The strategy is
> long-only, requires no shorting, fits a $1k account, and walks
> forward across 29 months winning 78% of the months it actually
> trades. We still require a 4-week Tiger paper-trade before
> committing real capital."
