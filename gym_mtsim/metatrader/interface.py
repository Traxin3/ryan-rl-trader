from typing import Any
from enum import Enum
from datetime import datetime

import numpy as np

try:
    import MetaTrader5 as mt5
    from MetaTrader5 import SymbolInfo as MtSymbolInfo
    MT5_AVAILABLE = True
except ImportError:
    MtSymbolInfo = object
    MT5_AVAILABLE = False


class Timeframe(Enum):
    M1 = 1  # mt5.TIMEFRAME_M1
    M2 = 2  # mt5.TIMEFRAME_M2
    M3 = 3  # mt5.TIMEFRAME_M3
    M4 = 4  # mt5.TIMEFRAME_M4
    M5 = 5  # mt5.TIMEFRAME_M5
    M6 = 6  # mt5.TIMEFRAME_M6
    M10 = 10  # mt5.TIMEFRAME_M10
    M12 = 12  # mt5.TIMEFRAME_M12
    M15 = 15  # mt5.TIMEFRAME_M15
    M20 = 20  # mt5.TIMEFRAME_M20
    M30 = 30  # mt5.TIMEFRAME_M30
    H1 = 1  | 0x4000  # mt5.TIMEFRAME_H1
    H2 = 2  | 0x4000  # mt5.TIMEFRAME_H2
    H4 = 4  | 0x4000  # mt5.TIMEFRAME_H4
    H3 = 3  | 0x4000  # mt5.TIMEFRAME_H3
    H6 = 6  | 0x4000  # mt5.TIMEFRAME_H6
    H8 = 8  | 0x4000  # mt5.TIMEFRAME_H8
    H12 = 12 | 0x4000  # mt5.TIMEFRAME_H12
    D1 = 24 | 0x4000  # mt5.TIMEFRAME_D1
    W1 = 1  | 0x8000  # mt5.TIMEFRAME_W1
    MN1 = 1  | 0xC000  # mt5.TIMEFRAME_MN1


_SOFT_FALLBACK = True  # allow running without MT5 (training/backtest environments)


def initialize() -> bool:
    if not MT5_AVAILABLE:
        if _SOFT_FALLBACK:
            return True
        _raise_mt5_unavailable()
    return mt5.initialize()


def shutdown() -> None:
    if not MT5_AVAILABLE:
        if _SOFT_FALLBACK:
            return None
        _raise_mt5_unavailable()
    mt5.shutdown()


def copy_rates_range(symbol: str, timeframe: Timeframe, date_from: datetime, date_to: datetime) -> np.ndarray:
    if not MT5_AVAILABLE:
        if _SOFT_FALLBACK:
            # Return an empty array with MT5-like dtype to avoid crashes in callers
            return np.empty((0,), dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')])
        _raise_mt5_unavailable()
    return mt5.copy_rates_range(symbol, timeframe.value, date_from, date_to)


def symbol_info(symbol: str) -> Any:
    if not MT5_AVAILABLE:
        if _SOFT_FALLBACK:
            class _MockInfo:
                def __init__(self, name):
                    self.name = name
                    self.trade_contract_size = 100000
                    self.margin_initial = 0.0
                    self.margin_maintenance = 0.0
                    self.trade_tick_value = 1.0
                    self.trade_tick_size = 0.0001
                    self.trade_mode = 0
                    self.spread = 10
                    self.point = 0.0001
                    self.digits = 5
            return _MockInfo(symbol)
        _raise_mt5_unavailable()
    return mt5.symbol_info(symbol)


def _raise_mt5_unavailable() -> None:
    raise OSError("MetaTrader5 is not available on your platform.")
