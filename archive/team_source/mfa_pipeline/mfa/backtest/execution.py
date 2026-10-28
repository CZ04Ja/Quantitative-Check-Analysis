"""Cost & execution model. Linear cost in turnover plus borrow on shorts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_bps: float = 1.0       # one-way, on traded notional
    slippage_bps: float = 2.0         # one-way, on traded notional
    borrow_cost_bps_annual: float = 50.0   # annual rate on the short notional

    @property
    def trade_cost_bps(self) -> float:
        """Round-trip cost is implicit in 'apply per-side on turnover'."""
        return self.commission_bps + self.slippage_bps

    def cost_for_step(self, turnover: float, short_notional_avg: float, days: int = 1) -> float:
        """Return the cost in dollars per $1 of capital for one step.

        turnover            = sum(|w_t - w_{t-1}|) (a fraction of capital)
        short_notional_avg  = average of sum(max(-w, 0)) over the step
        days                = bars in the step (for the borrow accrual)
        """
        trade = turnover * (self.trade_cost_bps / 1e4)
        borrow = short_notional_avg * (self.borrow_cost_bps_annual / 1e4) * (days / 252.0)
        return trade + borrow
