"""Classic technical indicators, vectorized over wide panels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder RSI in [0, 100]."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/window
    roll_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out


def macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> dict[str, pd.DataFrame]:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    line = fast_ema - slow_ema
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return {"macd": line, "signal": sig, "hist": hist}


def bollinger_bands(close: pd.DataFrame, window: int = 20, k: float = 2.0
                    ) -> dict[str, pd.DataFrame]:
    ma = close.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = close.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return {"middle": ma, "upper": upper, "lower": lower, "pct_b": pctb}


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    pc = close.shift(1)
    tr = pd.concat([
        (high - low).stack(),
        (high - pc).abs().stack(),
        (low - pc).abs().stack(),
    ], axis=1).max(axis=1).unstack()
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()
