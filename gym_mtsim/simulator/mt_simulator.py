from typing import List, Tuple, Dict, Any, Optional
import os
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import linregress
from ..metatrader import Timeframe, SymbolInfo, retrieve_data
from .order import OrderType, Order
from .exceptions import SymbolNotFound, OrderNotFound
from gym_mtsim.metatrader.api import fetch_mt5_rates, _get_symbol_info


class MtSimulator:
    def get_bos_choch_flags(self, symbol: str, tf: int, lookback: int = 3) -> np.ndarray:
        """
        Returns binary BOS/CHoCH flags for the last `lookback` candles.
        BOS (Break of Structure): 1 if price breaks previous high/low
        CHoCH (Change of Character): 1 if market shifts from bullish to bearish or vice versa
        
        Output: np.array of shape (lookback, 2) [BOS, CHoCH]
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, self.current_time, tf)
        current_idx = df.index.get_loc(nearest)
        start_idx = max(0, current_idx - lookback + 1)
        end_idx = current_idx + 1
        
        results = np.zeros((lookback, 2))
        
        for i in range(start_idx, end_idx):
            idx_in_results = i - start_idx
            if i < 2:  # Not enough data for analysis
                continue
                
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev_prev = df.iloc[i-2]
            
            bos = 0
            if current['High'] > prev['High'] and prev['High'] > prev_prev['High']:
                bos = 1  # Bullish BOS
            elif current['Low'] < prev['Low'] and prev['Low'] < prev_prev['Low']:
                bos = 1  # Bearish BOS
                
            choch = 0
            if (prev_prev['Close'] > prev_prev['Open'] and 
                prev['Close'] < prev['Open'] and 
                current['Close'] < current['Open']):
                choch = 1
            elif (prev_prev['Close'] < prev_prev['Open'] and 
                  prev['Close'] > prev['Open'] and 
                  current['Close'] > current['Open']):
                choch = 1
                
            results[idx_in_results] = [bos, choch]
            
        return results

    def get_trend_state(self, symbol: str, tf: int, window: int = 20) -> int:
        """
        Returns trend state: +1 (bull), -1 (bear), 0 (range)
        Uses linear regression on closing prices over the window period
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, self.current_time, tf)
        current_idx = df.index.get_loc(nearest)
        start_idx = max(0, current_idx - window + 1)
        
        closes = df.iloc[start_idx:current_idx+1]['Close'].values
        if len(closes) < 2:
            return 0
            
        x = np.arange(len(closes))
        slope, _, _, _, _ = linregress(x, closes)
        
        price_change_pct = (closes[-1] - closes[0]) / closes[0]
        
        if slope > 0 and price_change_pct > 0.005:  # Minimum 0.5% change for trend
            return 1  # Bullish
        elif slope < 0 and price_change_pct < -0.005:
            return -1  # Bearish
        return 0  # Range

    def get_weak_strong_high_low(self, symbol: str, tf: int) -> Tuple[bool, bool, bool, bool]:
        """
        Returns (weak_high, strong_high, weak_low, strong_low) as booleans.
        Weak high/low: Price touched but didn't close beyond
        Strong high/low: Price closed beyond the level
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, self.current_time, tf)
        current_idx = df.index.get_loc(nearest)
        if current_idx < 1:
            return (False, False, False, False)
            
        current = df.iloc[current_idx]
        prev = df.iloc[current_idx-1]
        
        weak_high = current['High'] > prev['High'] and current['Close'] < prev['High']
        strong_high = current['High'] > prev['High'] and current['Close'] > prev['High']
        weak_low = current['Low'] < prev['Low'] and current['Close'] > prev['Low']
        strong_low = current['Low'] < prev['Low'] and current['Close'] < prev['Low']
        
        return (weak_high, strong_high, weak_low, strong_low)

    def get_equal_high_low_count(self, symbol: str, tf: int, bars: int = 96) -> Tuple[int, int]:
        """
        Returns (equal_high_count, equal_low_count) in last `bars` candles.
        Equal highs/lows are counted when prices are within 0.05% of each other.
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, self.current_time, tf)
        current_idx = df.index.get_loc(nearest)
        start_idx = max(0, current_idx - bars + 1)
        
        highs = df.iloc[start_idx:current_idx+1]['High'].values
        lows = df.iloc[start_idx:current_idx+1]['Low'].values
        
        high_pct_diff = np.abs(np.diff(highs)) / highs[:-1]
        low_pct_diff = np.abs(np.diff(lows)) / lows[:-1]
        
        equal_high_count = np.sum(high_pct_diff < 0.0005)
        equal_low_count = np.sum(low_pct_diff < 0.0005)
        
        return (equal_high_count, equal_low_count)

    def get_session_info(self, dt: datetime) -> Tuple[np.ndarray, float, int, int]:
        """
        Returns (session_onehot[6], minutes_since_open, magic_manip_1, magic_manip_2)
        Sessions: Tokyo, London, NY, Sydney, Frankfurt, Overlap periods
        """
        hour = dt.hour
        minute = dt.minute
        weekday = dt.weekday()
        
        sessions = np.zeros(6)
        
        if 0 <= hour < 6:
            sessions[0] = 1
        elif 7 <= hour < 16:
            sessions[1] = 1
        elif 12 <= hour < 20:
            sessions[2] = 1
        elif 21 <= hour <= 23 or 0 <= hour < 5:
            sessions[3] = 1
        elif 6 <= hour < 17:
            sessions[4] = 1
            
        if (7 <= hour < 9) or (12 <= hour < 16):
            sessions[5] = 1
            
        if sessions[0]:  # Tokyo
            minutes_since_open = hour * 60 + minute
        elif sessions[1]:  # London
            minutes_since_open = (hour - 7) * 60 + minute
        elif sessions[2]:  # NY
            minutes_since_open = (hour - 12) * 60 + minute
        elif sessions[3]:  # Sydney
            minutes_since_open = ((hour - 21) % 24) * 60 + minute
        else:
            minutes_since_open = 0
            
        magic_manip_1 = (hour * 60 + minute) % 144
        magic_manip_2 = (weekday * 24 + hour) % 89
        
        return sessions, minutes_since_open, magic_manip_1, magic_manip_2

    def get_liquidity_features(self, symbol: str, tf: int, dt: datetime) -> Tuple[float, int, int, float]:
        """
        Returns (distance_to_liquidity_pool, liquidity_sweep_flag, liquidity_buildup, order_block_strength)
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, dt, tf)
        current_idx = df.index.get_loc(nearest)
        if current_idx < 20:  # Need enough data
            return 0.0, 0, 0, 0.0
            
        highs = df.iloc[current_idx-20:current_idx+1]['High'].values
        lows = df.iloc[current_idx-20:current_idx+1]['Low'].values
        
        high_counts = {}
        low_counts = {}
        bin_size = np.mean(highs - lows) * 0.1  # 10% of average range
        
        for h in highs:
            if bin_size > 1e-8:
                bin_val = round(h / bin_size) * bin_size
                high_counts[bin_val] = high_counts.get(bin_val, 0) + 1
        for l in lows:
            if bin_size > 1e-8:
                bin_val = round(l / bin_size) * bin_size
                low_counts[bin_val] = low_counts.get(bin_val, 0) + 1
            
        current_price = df.iloc[current_idx]['Close']
        nearest_high_pool = min(high_counts.keys(), key=lambda x: abs(x - current_price)) if high_counts else 0
        nearest_low_pool = min(low_counts.keys(), key=lambda x: abs(x - current_price)) if low_counts else 0
        
        distance_to_pool = min(
            abs(current_price - nearest_high_pool),
            abs(current_price - nearest_low_pool)
        ) / current_price
        
        sweep_flag = 0
        if current_idx > 2:
            prev = df.iloc[current_idx-1]
            prev_prev = df.iloc[current_idx-2]
            current = df.iloc[current_idx]
            
            if (current['Low'] < prev['Low'] and 
                current['Close'] > prev['Low'] and 
                prev['Low'] < prev_prev['Low']):
                sweep_flag = 1
            elif (current['High'] > prev['High'] and 
                  current['Close'] < prev['High'] and 
                  prev['High'] > prev_prev['High']):
                sweep_flag = -1
                
        buildup = 0
        if current_idx > 5:
            recent_highs = highs[-5:]
            recent_lows = lows[-5:]
            
            if len(set(np.round(recent_highs / bin_size))) < 3:
                buildup += 1
            if len(set(np.round(recent_lows / bin_size))) < 3:
                buildup += 1
                
        ob_strength = 0.0
        if current_idx > 3:
            prev_candles = df.iloc[current_idx-3:current_idx+1]
            impulsive_moves = 0
            for _, row in prev_candles.iterrows():
                rng = row['High'] - row['Low']
                if rng > 1e-8 and (row['Close'] - row['Open']) / rng > 0.7:
                    impulsive_moves += 1
            ob_strength = impulsive_moves / 4.0
            
        return distance_to_pool, sweep_flag, buildup, ob_strength

    def get_cycle_state(self, symbol: str, tf: int, dt: datetime) -> int:
        """
        Returns LIT cycle state: 0=Build-up, 1=Inducement, 2=BOS, 3=Mitigation
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, dt, tf)
        current_idx = df.index.get_loc(nearest)
        if current_idx < 10:
            return 0
            
        recent = df.iloc[current_idx-10:current_idx+1]
        highs = recent['High'].values
        lows = recent['Low'].values
        closes = recent['Close'].values
        
        atr = np.mean(highs - lows)
        trend = self.get_trend_state(symbol, tf, 10)
        
        if atr < np.mean(highs) * 0.002 and trend == 0:
            return 0
            
        if (atr > np.mean(highs) * 0.003 and 
            any(closes[i] > highs[i-1] or closes[i] < lows[i-1] for i in range(1, len(closes)))):
            return 1
            
        if (trend != 0 and 
            ((trend == 1 and all(closes[i] > closes[i-1] for i in range(1, 4))) or
             (trend == -1 and all(closes[i] < closes[i-1] for i in range(1, 4))))):
            return 2
            
        return 3

    def get_trade_risk_features(self, symbol: str, tf: int, dt: datetime) -> Tuple[float, int, float, int, int]:
        """
        Returns (rr_potential, stop_hunt_flag, atr15, london_bias, ny_bias)
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, dt, tf)
        current_idx = df.index.get_loc(nearest)
        if current_idx < 15:
            return 0.0, 0, 0.0, 0, 0
            
        current = df.iloc[current_idx]
        recent = df.iloc[current_idx-15:current_idx+1]
        
        atr = np.mean(recent['High'].values - recent['Low'].values)
        support = min(recent['Low'].values)
        resistance = max(recent['High'].values)
        
        rr_potential = min(
            (resistance - current['Close']) / atr,
            (current['Close'] - support) / atr
        )
        
        stop_hunt_flag = 0
        if current_idx > 3:
            prev = df.iloc[current_idx-1]
            if (current['High'] > resistance and current['Close'] < prev['Close']):
                stop_hunt_flag = 1
            elif (current['Low'] < support and current['Close'] > prev['Close']):
                stop_hunt_flag = -1
                
        london_bias = 0
        ny_bias = 0
        if current_idx > 100:  # Need enough data for statistical analysis
            session_data = []
            for i in range(current_idx-100, current_idx+1):
                row = df.iloc[i]
                hour = row.name.hour
                session = 'London' if 7 <= hour < 16 else 'NY' if 12 <= hour < 20 else None
                if session:
                    session_data.append((session, row['Close'] - row['Open']))
                    
            if session_data:
                london_returns = [x[1] for x in session_data if x[0] == 'London']
                ny_returns = [x[1] for x in session_data if x[0] == 'NY']
                
                if london_returns:
                    london_bias = 1 if np.mean(london_returns) > 0 else -1
                if ny_returns:
                    ny_bias = 1 if np.mean(ny_returns) > 0 else -1
                    
        return rr_potential, stop_hunt_flag, atr, london_bias, ny_bias

    def get_price_action_features(self, symbol: str, tf: int, dt: datetime) -> Tuple[int, float, float, float]:
        """
        Returns (vector_candle_flag, impulse_vs_retrace, candle_speed, imbalance_size)
        """
        df = self.symbols_data[(symbol, tf)]
        nearest = self.nearest_time(symbol, dt, tf)
        current_idx = df.index.get_loc(nearest)
        if current_idx < 1:
            return 0, 0.0, 0.0, 0.0
            
        current = df.iloc[current_idx]
        prev = df.iloc[current_idx-1]
        
        vector_candle_flag = 0
        body_size = abs(current['Close'] - current['Open'])
        total_range = current['High'] - current['Low']
        if total_range > 1e-8:
            body_ratio = body_size / total_range
            if body_ratio > 0.7:  # Mostly body with small wicks
                vector_candle_flag = 1 if current['Close'] > current['Open'] else -1
                
        impulse_vs_retrace = 0.0
        if current_idx > 5:
            prev_trend = linregress(
                np.arange(5),
                df.iloc[current_idx-5:current_idx]['Close'].values
            ).slope
            
            current_direction = np.sign(current['Close'] - prev['Close'])
            if prev_trend * current_direction > 0:
                impulse_vs_retrace = 1.0  # Impulse
            else:
                impulse_vs_retrace = -1.0  # Retrace
                
        if current_idx > 10:
            avg_body = np.mean([
                abs(row['Close'] - row['Open']) 
                for _, row in df.iloc[current_idx-10:current_idx].iterrows()
            ])
            candle_speed = body_size / (avg_body + 1e-6)
        else:
            candle_speed = 0.0
            
        imbalance_size = 0.0
        if current['Low'] > prev['High']:  # Gap up
            imbalance_size = (current['Low'] - prev['High']) / prev['High']
        elif current['High'] < prev['Low']:  # Gap down
            imbalance_size = (prev['Low'] - current['High']) / prev['Low']
            
        return vector_candle_flag, impulse_vs_retrace, candle_speed, imbalance_size

    def __init__(
        self,
        unit: str = 'USD',
        balance: float = 10000.,
        leverage: float = 100.,
        stop_out_level: float = 0.2,
        hedge: bool = True,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[int]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> None:
        self.unit = unit
        self.balance = balance
        self.equity = balance
        self.leverage = leverage
        self.stop_out_level = stop_out_level
        self.hedge = hedge
        self.margin = 0.
        self.symbols_info: Dict[str, SymbolInfo] = {}
        self.symbols_data: Dict[Tuple[str, int], pd.DataFrame] = {}
        self.orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.current_time: datetime = NotImplemented
        self.symbols = symbols or []
        self.timeframes = timeframes or []
        self.start = start
        self.end = end
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
        os.makedirs(data_dir, exist_ok=True)
        total_pairs = len(self.symbols) * len(self.timeframes)
        local_pairs = []
        download_pairs = []
        for symbol in self.symbols:
            for tf in self.timeframes:
                start_str = self.start.strftime('%Y%m%d') if self.start else 'none'
                end_str = self.end.strftime('%Y%m%d') if self.end else 'none'
                cache_file = os.path.join(data_dir, f"{symbol}_{tf}_{start_str}_{end_str}.pkl")
                if os.path.exists(cache_file):
                    local_pairs.append((symbol, tf, cache_file))
                else:
                    download_pairs.append((symbol, tf, cache_file))

        if local_pairs:
            with tqdm(total=len(local_pairs), desc="Loading local market data", unit="pair") as pbar:
                for symbol, tf, cache_file in local_pairs:
                    try:
                        self.symbols_info[symbol] = _get_symbol_info(symbol)
                        df = pd.read_pickle(cache_file)
                        self.symbols_data[(symbol, tf)] = df
                        pbar.update(1)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load local data for {symbol}: {e}")

        if download_pairs:
            with tqdm(total=len(download_pairs), desc="Downloading market data", unit="pair") as pbar:
                for symbol, tf, cache_file in download_pairs:
                    try:
                        self.symbols_info[symbol] = _get_symbol_info(symbol)
                        df = fetch_mt5_rates(symbol, tf, self.start, self.end)
                        df.to_pickle(cache_file)
                        self.symbols_data[(symbol, tf)] = df
                        pbar.update(1)
                    except Exception as e:
                        raise RuntimeError(f"Failed to download data for {symbol}: {e}")

    @property
    def free_margin(self) -> float:
        return self.equity - self.margin

    @property
    def margin_level(self) -> float:
        margin = round(self.margin, 6)
        if margin == 0.:
            return float('inf')
        return self.equity / margin

    def download_data(
        self, symbols: List[str], timeframes: List[int], start: datetime, end: datetime
    ) -> None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
        os.makedirs(data_dir, exist_ok=True)
        total_downloads = len(symbols) * len(timeframes)
        with tqdm(total=total_downloads, desc="Downloading market data", unit="pair") as pbar:
            for symbol in symbols:
                for tf in timeframes:
                    cache_file = os.path.join(data_dir, f"{symbol}_{tf}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.pkl")
                    if os.path.exists(cache_file):
                        df = pd.read_pickle(cache_file)
                        si = _get_symbol_info(symbol)
                    else:
                        si, df = retrieve_data(symbol, start, end, Timeframe(f"M{tf}"))
                        df.to_pickle(cache_file)
                    self.symbols_info[symbol] = si
                    self.symbols_data[(symbol, tf)] = df
                    pbar.update(1)

    def tick(self, delta_time: timedelta = timedelta()) -> None:
        self._check_current_time()

        self.current_time += delta_time
        self.equity = self.balance

        for order in self.orders:
            order.exit_time = self.current_time
            order.exit_price = self.price_at(order.symbol, order.exit_time)['Close']
            self._update_order_profit(order)
            self.equity += order.profit

        while self.margin_level < self.stop_out_level and len(self.orders) > 0:
            most_unprofitable_order = min(self.orders, key=lambda order: order.profit)
            self.close_order(most_unprofitable_order)

        if self.balance < 0.:
            self.balance = 0.
            self.equity = self.balance

    def nearest_time(self, symbol: str, time: datetime, tf: int = None) -> datetime:
        if tf is None:
            tf = self.timeframes[0]
        df = self.symbols_data[(symbol, tf)]
        if time in df.index:
            return time
        try:
            i, = df.index.get_indexer([time], method='ffill')
        except KeyError:
            i, = df.index.get_indexer([time], method='bfill')
        return df.index[i]

    def price_at(self, symbol: str, time: datetime, tf: int = None) -> pd.Series:
        if tf is None:
            tf = self.timeframes[0]
        df = self.symbols_data[(symbol, tf)]
        time = self.nearest_time(symbol, time, tf)
        return df.loc[time]

    def symbol_orders(self, symbol: str) -> List[Order]:
        return [order for order in self.orders if order.symbol == symbol]

    def create_order(
        self, order_type: OrderType, symbol: str, volume: float, fee: float = 0.0005,
        raise_exception: bool = True
    ) -> Optional[Order]:
        self._check_current_time()
        self._check_volume(symbol, volume)
        if fee < 0.:
            raise ValueError(f"negative fee '{fee}'")

        if self.hedge:
            return self._create_hedged_order(order_type, symbol, volume, fee, raise_exception)
        return self._create_unhedged_order(order_type, symbol, volume, fee, raise_exception)

    def _create_hedged_order(
        self, order_type: OrderType, symbol: str, volume: float, fee: float,
        raise_exception: bool
    ) -> Optional[Order]:
        order_id = len(self.closed_orders) + len(self.orders) + 1
        entry_time = self.current_time
        entry_price = self.price_at(symbol, entry_time)['Close']
        exit_time = entry_time
        exit_price = entry_price

        order = Order(
            order_id, order_type, symbol, volume, fee,
            entry_time, entry_price, exit_time, exit_price
        )
        self._update_order_profit(order)
        self._update_order_margin(order)

        if order.margin > self.free_margin + order.profit:
            if raise_exception:
                raise ValueError(
                    f"low free margin (order margin={order.margin}, order profit={order.profit}, "
                    f"free margin={self.free_margin})"
                )
            return None

        self.equity += order.profit
        self.margin += order.margin
        self.orders.append(order)
        return order

    def _create_unhedged_order(
        self, order_type: OrderType, symbol: str, volume: float, fee: float,
        raise_exception: bool
    ) -> Optional[Order]:
        if symbol not in map(lambda order: order.symbol, self.orders):
            return self._create_hedged_order(order_type, symbol, volume, fee, raise_exception)

        old_order: Order = self.symbol_orders(symbol)[0]

        if old_order.type == order_type:
            new_order = self._create_hedged_order(order_type, symbol, volume, fee, raise_exception)
            if new_order is None:
                return None
            self.orders.remove(new_order)

            entry_price_weighted_average = np.average(
                [old_order.entry_price, new_order.entry_price],
                weights=[old_order.volume, new_order.volume]
            )

            old_order.volume += new_order.volume
            old_order.profit += new_order.profit
            old_order.margin += new_order.margin
            old_order.entry_price = entry_price_weighted_average
            old_order.fee = max(old_order.fee, new_order.fee)

            return old_order

        if volume >= old_order.volume:
            self.close_order(old_order)
            if volume > old_order.volume:
                return self._create_hedged_order(order_type, symbol, volume - old_order.volume, fee)
            return old_order

        partial_profit = (volume / old_order.volume) * old_order.profit
        partial_margin = (volume / old_order.volume) * old_order.margin

        old_order.volume -= volume
        old_order.profit -= partial_profit
        old_order.margin -= partial_margin

        self.balance += partial_profit
        self.margin -= partial_margin

        return old_order

    def close_order(self, order: Order) -> float:
        self._check_current_time()
        if order not in self.orders:
            raise OrderNotFound("order not found in the order list")

        order.exit_time = self.current_time
        order.exit_price = self.price_at(order.symbol, order.exit_time)['Close']
        self._update_order_profit(order)

        self.balance += order.profit
        self.margin -= order.margin

        order.exit_balance = self.balance
        order.exit_equity = self.equity

        order.closed = True
        self.orders.remove(order)
        self.closed_orders.append(order)

        return order.profit

    def get_state(self) -> Dict[str, Any]:
        orders = []
        for order in reversed(self.closed_orders + self.orders):
            orders.append({
                'Id': order.id,
                'Symbol': order.symbol,
                'Type': order.type.name,
                'Volume': order.volume,
                'Entry Time': order.entry_time,
                'Entry Price': order.entry_price,
                'Exit Time': order.exit_time,
                'Exit Price': order.exit_price,
                'Exit Balance': order.exit_balance,
                'Exit Equity': order.exit_equity,
                'Profit': order.profit,
                'Margin': order.margin,
                'Fee': order.fee,
                'Closed': order.closed,
            })
        orders_df = pd.DataFrame(orders)

        return {
            'current_time': self.current_time,
            'balance': self.balance,
            'equity': self.equity,
            'margin': self.margin,
            'free_margin': self.free_margin,
            'margin_level': self.margin_level,
            'orders': orders_df,
        }

    def _update_order_profit(self, order: Order) -> None:
        diff = order.exit_price - order.entry_price
        v = order.volume * self.symbols_info[order.symbol].trade_contract_size
        local_profit = v * (order.type.sign * diff - order.fee)
        order.profit = local_profit * self._get_unit_ratio(order.symbol, order.exit_time)

    def _update_order_margin(self, order: Order) -> None:
        v = order.volume * self.symbols_info[order.symbol].trade_contract_size
        local_margin = (v * order.entry_price) / self.leverage
        local_margin *= self.symbols_info[order.symbol].margin_rate
        order.margin = local_margin * self._get_unit_ratio(order.symbol, order.entry_time)

    def _get_unit_ratio(self, symbol: str, time: datetime) -> float:
        symbol_info = self.symbols_info[symbol]
        if self.unit == symbol_info.currency_profit:
            return 1.

        if self.unit == symbol_info.currency_margin:
            return 1 / self.price_at(symbol, time)['Close']

        currency = symbol_info.currency_profit
        unit_symbol_info = self._get_unit_symbol_info(currency)
        if unit_symbol_info is None:
            raise SymbolNotFound(f"unit symbol for '{currency}' not found")

        unit_price = self.price_at(unit_symbol_info.name, time)['Close']
        if unit_symbol_info.currency_margin == self.unit:
            unit_price = 1. / unit_price

        return unit_price

    def _get_unit_symbol_info(self, currency: str) -> Optional[SymbolInfo]:
        for info in self.symbols_info.values():
            if currency in info.currencies and self.unit in info.currencies:
                return info
        return None

    def _check_current_time(self) -> None:
        if self.current_time is NotImplemented:
            raise ValueError("'current_time' must have a value")

    def _check_volume(self, symbol: str, volume: float) -> None:
        symbol_info = self.symbols_info[symbol]

        if not (symbol_info.volume_min <= volume <= symbol_info.volume_max):
            raise ValueError(
                f"'volume' must be in range [{symbol_info.volume_min}, {symbol_info.volume_max}]"
            )

        if not round(volume / symbol_info.volume_step, 6).is_integer():
            raise ValueError(f"'volume' must be a multiple of {symbol_info.volume_step}")
