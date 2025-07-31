from typing import Tuple

import pytz
import calendar
from datetime import datetime, timedelta

import pandas as pd
import MetaTrader5 as mt5

from . import interface as mt
from .symbol import SymbolInfo


def retrieve_data(
        symbol: str, from_dt: datetime, to_dt: datetime, timeframe: mt.Timeframe
    ) -> Tuple[SymbolInfo, pd.DataFrame]:

    if not mt.initialize():
        raise ConnectionError(f"MetaTrader cannot be initialized")

    symbol_info = _get_symbol_info(symbol)

    utc_from = _local2utc(from_dt)
    utc_to = _local2utc(to_dt)
    all_rates = []

    partial_from = utc_from
    partial_to = _add_months(partial_from, 1)

    while partial_from < utc_to:
        rates = mt.copy_rates_range(symbol, timeframe, partial_from, partial_to)
        all_rates.extend(rates)
        partial_from = _add_months(partial_from, 1)
        partial_to = min(_add_months(partial_to, 1), utc_to)

    all_rates = [list(r) for r in all_rates]

    rates_frame = pd.DataFrame(
        all_rates,
        columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', '_', '_'],
    )
    rates_frame['Time'] = pd.to_datetime(rates_frame['Time'], unit='s', utc=True)

    data = rates_frame[['Time', 'Open', 'Close', 'Low', 'High', 'Volume']].set_index('Time')
    data = data.loc[~data.index.duplicated(keep='first')]

    mt.shutdown()

    return symbol_info, data


def _get_symbol_info(symbol: str) -> SymbolInfo:
    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5 initialize() failed: {mt5.last_error()}")
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"MetaTrader5 symbol_select({symbol}) failed: {mt5.last_error()}")
    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise ValueError(f"MetaTrader5 symbol_info({symbol}) returned None. Make sure the symbol is visible in Market Watch.")
    symbol_info = SymbolInfo(info)
    mt5.shutdown()
    return symbol_info


def _local2utc(dt: datetime) -> datetime:
    return dt.astimezone(pytz.timezone('Etc/UTC'))


def _add_months(sourcedate: datetime, months: int) -> datetime:
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])

    return datetime(
        year, month, day,
        sourcedate.hour, sourcedate.minute, sourcedate.second,
        tzinfo=sourcedate.tzinfo
    )


def fetch_mt5_rates(symbol: str, timeframe: int, start: datetime = None, end: datetime = None) -> pd.DataFrame:
    """
    Fetch historical rates from MetaTrader5 for a given symbol, timeframe, and date range (tries 365, 180, 90, 30, 7, 1 days).
    Returns a pandas DataFrame indexed by datetime, or raises ValueError if no data is returned.
    """
    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5 initialize() failed: {mt5.last_error()}")
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"MetaTrader5 symbol_select({symbol}) failed: {mt5.last_error()}")
    tf_map = {1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15, 30: mt5.TIMEFRAME_M30,
              60: mt5.TIMEFRAME_H1, 240: mt5.TIMEFRAME_H4, 1440: mt5.TIMEFRAME_D1}
    if timeframe not in tf_map:
        mt5.shutdown()
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    tf_const = tf_map[timeframe]
    now = datetime.now()
    if end is None or start is None or start >= end:
        end = now
    day_windows = [365, 180, 90, 30, 7, 1]
    for days in day_windows:
        s = end - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, tf_const, s, end)
        if rates is not None and len(rates) > 0:
            mt5.shutdown()
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'TickVolume',
                'spread': 'Spread',
                'real_volume': 'RealVolume'
            }, inplace=True)
            return df
    mt5.shutdown()
    raise ValueError(f"No data returned for {symbol} {timeframe}m in the last {day_windows[0]} to {day_windows[-1]} days up to {end}")
