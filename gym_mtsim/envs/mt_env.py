from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from .optimized_feature_engine import OptimizedFeatureEngine
import copy
from datetime import datetime
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_colors
import plotly.graph_objects as go
import gymnasium as gym
from gymnasium import spaces
from ..simulator import MtSimulator, OrderType
from collections import deque, defaultdict
import time
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardMetricsCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.trade_wins = 0
        self.trade_losses = 0
        self.trade_profits = []
        self.equity_peak = -np.inf
        self.equity_drawdown = 0
        self.position_holding_times = []
        self.current_positions = {}
        self.episode_start_time = time.time()
        self.portfolio_returns = []
        self.episode_reward_components = defaultdict(list)
        self.timeframe_factors = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.trade_profits) > 0:
            total_trades = self.trade_wins + self.trade_losses
            win_rate = self.trade_wins / (total_trades + 1e-6)
            avg_profit = np.mean(self.trade_profits)
            avg_holding_time = np.mean(self.position_holding_times) if self.position_holding_times else 0
            
            returns = np.array(self.trade_profits)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252) if len(returns) > 1 else 0
            sortino = self._calculate_sortino_ratio(returns)
            
            self.logger.record("metrics/win_rate", win_rate)
            self.logger.record("metrics/avg_profit_per_trade", avg_profit)
            self.logger.record("metrics/max_drawdown", self.equity_drawdown)
            self.logger.record("metrics/sharpe_ratio", sharpe)
            self.logger.record("metrics/sortino_ratio", sortino)
            self.logger.record("metrics/avg_holding_time", avg_holding_time)
            self.logger.record("metrics/active_positions", len(self.current_positions))
            self.logger.record("metrics/total_trades", total_trades)
            self.logger.record("metrics/profit_factor", self._calculate_profit_factor())
            
            self.logger.record("reward_components/equity_change", np.mean(self.episode_reward_components['equity_change']))
            self.logger.record("reward_components/trade_quality", np.mean(self.episode_reward_components['trade_quality']))
            self.logger.record("reward_components/risk_penalty", np.mean(self.episode_reward_components['risk_penalty']))
            self.logger.record("reward_components/consistency_bonus", np.mean(self.episode_reward_components['consistency_bonus']))
            self.logger.record("reward_components/timeframe_factor", np.mean(self.timeframe_factors))
            
        self.trade_profits = []
        self.position_holding_times = []
        self.episode_start_time = time.time()
        self.episode_reward_components = defaultdict(list)
        self.timeframe_factors = []

    def _calculate_sortino_ratio(self, returns: np.ndarray, annualize_factor=252) -> float:
        if len(returns) < 2:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.mean(returns) * annualize_factor / (np.std(returns) + 1e-6)
        downside_std = np.std(downside_returns)
        return np.mean(returns) * np.sqrt(annualize_factor) / (downside_std + 1e-6)

    def _calculate_profit_factor(self) -> float:
        if not self.trade_profits:
            return 0.0
        gross_profit = sum(p for p in self.trade_profits if p > 0)
        gross_loss = abs(sum(p for p in self.trade_profits if p < 0))
        return gross_profit / (gross_loss + 1e-6)

    def update_trade_metrics(self, order_info: Dict) -> None:
        profit = order_info.get('profit', 0)
        if profit > 0:
            self.trade_wins += 1
        else:
            self.trade_losses += 1
        self.trade_profits.append(profit)
        self.portfolio_returns.append(profit)

    def update_drawdown(self, equity: float) -> None:
        self.equity_peak = max(self.equity_peak, equity)
        self.equity_drawdown = max(self.equity_drawdown, (self.equity_peak - equity) / (self.equity_peak + 1e-6))

    def update_position_times(self, current_orders: List) -> None:
        now = time.time()
        active_ids = {order.id for order in current_orders}
        
        for order_id in list(self.current_positions.keys()):
            if order_id not in active_ids:
                self.position_holding_times.append(now - self.current_positions.pop(order_id))
        
        for order in current_orders:
            if order.id not in self.current_positions:
                self.current_positions[order.id] = now

class MtEnv(gym.Env):
    metadata = {'render_modes': ['human', 'simple_figure', 'advanced_figure']}

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[int],
        window_size: int = 10,
        time_start: datetime = None,
        time_end: datetime = None,
        hold_threshold: float = 0.5,
        close_threshold: float = 0.5,
        fee: Union[float, Callable[[str], float]] = 0.0005,
        symbol_max_orders: int = 1,
        multiprocessing_processes: Optional[int] = None,
        render_mode: Optional[str] = None,
        reward_scaling: float = 1.0,
        max_leverage: float = 10.0,
        min_reward: float = -5.0,
        max_reward: float = 5.0,
        survival_bonus: float = 0.01,
        leverage_penalty: float = 0.01,
        trade_reward_multiplier: float = 1.0,
        risk_adjusted_reward: bool = True,
        drawdown_penalty: float = 0.1,
        early_close_bonus: float = 0.05,
        diversification_bonus: float = 0.02,
        volatility_penalty: float = 0.02,
        position_size_penalty: float = 0.01,
        min_tp_sl_ratio: float = 1.5,  # Minimum TP/SL ratio
        atr_multiplier: float = 2.0     # ATR multiplier for SL
    ) -> None:
        self.symbols = symbols
        self.timeframes = timeframes
        self.window_size = window_size
        self.hold_threshold = hold_threshold
        self.close_threshold = close_threshold
        self.fee = fee
        self.symbol_max_orders = symbol_max_orders
        self.reward_scaling = reward_scaling
        self.max_leverage = max_leverage
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.survival_bonus = survival_bonus
        self.leverage_penalty = leverage_penalty
        self.trade_reward_multiplier = trade_reward_multiplier
        self.risk_adjusted_reward = risk_adjusted_reward
        self.drawdown_penalty = drawdown_penalty
        self.early_close_bonus = early_close_bonus
        self.diversification_bonus = diversification_bonus
        self.volatility_penalty = volatility_penalty
        self.position_size_penalty = position_size_penalty
        self.min_tp_sl_ratio = min_tp_sl_ratio
        self.atr_multiplier = atr_multiplier

        self.multiprocessing_pool = Pool(multiprocessing_processes) if multiprocessing_processes else None
        self.render_mode = render_mode
        
        self.simulator = MtSimulator(
            symbols=symbols,
            timeframes=timeframes,
            start=time_start,
            end=time_end,
        )
        
        self.time_points = self.simulator.symbols_data[(symbols[0], timeframes[0])].index.to_pydatetime().tolist()
        self.simulator.current_time = self.time_points[0]
        self.prices = self._get_prices()
        self.optimized_feature_engine = OptimizedFeatureEngine(
            n_pca_components=10,
            variance_threshold=1e-5,
            cache_path=f"feature_cache_{'_'.join(self.symbols)}_{'_'.join(map(str, self.timeframes))}.pkl"
        )
        self.signal_features = self._process_data()
        self.features_shape = (window_size, self.signal_features.shape[1])
        
        INF = 1e10
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float64,
            shape=(len(self.symbols) * (self.symbol_max_orders + 2),))
        
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'equity': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'margin': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'features': spaces.Box(low=-INF, high=INF, shape=self.features_shape, dtype=np.float64),
            'orders': spaces.Box(
                low=-INF, high=INF, dtype=np.float64,
                shape=(len(self.symbols), self.symbol_max_orders, 3)
            )
        })
        
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.time_points) - 1
        self._truncated = False
        self._current_tick = self._start_tick
        self.history = []
        self.metrics_callback = TensorboardMetricsCallback()
        
        self.trade_history = []
        self.peak_equity = self.simulator.balance
        self.max_drawdown = 0.0
        self.position_start_times = {}
        self.portfolio_values = []
        self.returns_volatility = 0.0
        self.episode_reward_components = defaultdict(list)
        self.timeframe_factors = []

        if len(self.time_points) <= window_size:
            raise ValueError(f"Not enough data. Need {window_size+1} bars, got {len(self.time_points)}")

    def reset(self, seed=None, options=None) -> Dict[str, np.ndarray]:
        super().reset(seed=seed, options=options)
        self._truncated = False
        self._current_tick = self._start_tick
        self.simulator = copy.deepcopy(self.simulator)
        self.simulator.current_time = self.time_points[self._current_tick]
        self.history = [self._create_info()]
        
        self.trade_history = []
        self.peak_equity = self.simulator.balance
        self.max_drawdown = 0.0
        self.position_start_times = {}
        self.portfolio_values = [self.simulator.equity]
        self.returns_volatility = 0.0
        self.episode_reward_components = defaultdict(list)
        self.timeframe_factors = []
        
        return self._get_observation(), self._create_info()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        orders_info, closed_orders_info = self._apply_action(action)
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._truncated = True
        
        dt = self.time_points[self._current_tick] - self.time_points[self._current_tick - 1]
        self.simulator.tick(dt)
        
        self.portfolio_values.append(self.simulator.equity)
        if len(self.portfolio_values) > 2:
            returns = np.diff(np.log(self.portfolio_values))
            self.returns_volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        step_reward = self._calculate_reward(orders_info, closed_orders_info)
        
        for symbol, closed_orders in closed_orders_info.items():
            for order in closed_orders:
                self.trade_history.append(order)
                self.metrics_callback.update_trade_metrics(order)
        
        self._update_position_times()
        self.metrics_callback.update_position_times(self.simulator.orders)
        
        self._update_drawdown_metrics()
        self.metrics_callback.update_drawdown(self.simulator.equity)
        
        info = self._create_info(
            orders=orders_info, 
            closed_orders=closed_orders_info, 
            step_reward=step_reward,
            leverage=self.simulator.margin / (self.simulator.balance + 1e-6),
            drawdown=self.max_drawdown,
            win_rate=self._calculate_win_rate(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            sortino_ratio=self._calculate_sortino_ratio(),
            profit_factor=self._calculate_profit_factor(),
            volatility=self.returns_volatility
        )
        
        observation = self._get_observation()
        self.history.append(info)
        
        return observation, step_reward, False, self._truncated, info

    def _calculate_reward(self, orders_info: Dict, closed_orders_info: Dict) -> float:
        prev_info = self.history[-1]
        current_equity = self.simulator.equity
        
        equity_change = (current_equity - prev_info['equity']) / (prev_info['equity'] + 1e-6)
        self.episode_reward_components['equity_change'].append(equity_change)
        
        trade_quality = 0
        for symbol, closed_orders in closed_orders_info.items():
            for order in closed_orders:
                roi = order['profit'] / (order['margin'] + 1e-6)
                trade_duration = order.get('holding_time', 1.0)  # in hours
                
                if trade_duration > 0:
                    try:
                        exponent = np.log1p(roi) * (24*365/trade_duration)
                        exponent = np.clip(exponent, -700, 700)
                        annualized_roi = np.exp(exponent) - 1
                    except:
                        annualized_roi = 0  # Fallback for any numerical errors
                else:
                    annualized_roi = 0

                if len(self.trade_history) > 5:
                    recent_returns = np.array([t['profit']/t['margin'] for t in self.trade_history[-5:] if 'margin' in t])
                    if len(recent_returns) > 1:
                        sharpe_like = roi / (np.std(recent_returns) + 1e-6)
                        trade_quality += 0.5 * np.sign(roi) * np.log1p(abs(sharpe_like))
                
                trade_quality += 0.5 * np.sign(roi) * np.log1p(abs(annualized_roi))
                
                if roi > 0.01 and 'holding_time' in order:
                    trade_quality += self.early_close_bonus * min(1.0, 1.0 / (order['holding_time'] + 1e-6))
        
        margin_level = self.simulator.margin_level
        risk_penalty = -0.3 * (10.0 - min(margin_level, 10.0)) ** 0.5 if margin_level < 10.0 else 0
        self.episode_reward_components['risk_penalty'].append(risk_penalty)
        
        open_positions = len(self.simulator.orders)
        position_bonus = 0.05 * min(open_positions, len(self.symbols))
        
        unique_symbols = len({order.symbol for order in self.simulator.orders})
        diversification = self.diversification_bonus * unique_symbols / len(self.symbols)
        
        drawdown_penalty = -self.drawdown_penalty * (self.max_drawdown ** 1.5)
        volatility_penalty = -self.volatility_penalty * self.returns_volatility * 100
        
        total_margin = sum(order.margin for order in self.simulator.orders)
        position_size_penalty = -self.position_size_penalty * (total_margin / (self.simulator.balance + 1e-6))
        
        survival = self.survival_bonus if not self._truncated else 0
        leverage = self.simulator.margin / (self.simulator.balance + 1e-6)
        leverage_penalty = -self.leverage_penalty * (max(leverage - 1, 0) ** 2)
        
        consistency_bonus = 0
        if len(self.trade_history) >= 3:
            last_three = [t['profit'] for t in self.trade_history[-3:]]
            if all(p > 0 for p in last_three):
                min_profit = min(abs(p) for p in last_three)
                consistency_bonus = 0.5 * min_profit / (self.simulator.equity + 1e-6)
        self.episode_reward_components['consistency_bonus'].append(consistency_bonus)
        
        timeframe_factor = self._get_timeframe_factor()
        self.timeframe_factors.append(timeframe_factor)
        self.metrics_callback.timeframe_factors.append(timeframe_factor)
        
        volatility_factor = min(1.0, max(0.1, self.returns_volatility * 10))
        weights = {
            'equity_change': 0.3 * volatility_factor,
            'trade_quality': 0.25 * (2 - volatility_factor),
            'risk_penalty': 0.15,
            'position_bonus': 0.05,
            'diversification': 0.05,
            'drawdown_penalty': 0.05 + 0.05 * (1 - volatility_factor),
            'volatility_penalty': 0.05 * volatility_factor,
            'position_size_penalty': 0.03,
            'survival': 0.02,
            'leverage_penalty': 0.02,
            'consistency_bonus': 0.03
        }
        
        total_reward = (
            weights['equity_change'] * equity_change + 
            weights['trade_quality'] * trade_quality * self.trade_reward_multiplier + 
            weights['risk_penalty'] * risk_penalty + 
            weights['position_bonus'] * position_bonus +
            weights['diversification'] * diversification +
            weights['drawdown_penalty'] * drawdown_penalty +
            weights['volatility_penalty'] * volatility_penalty +
            weights['position_size_penalty'] * position_size_penalty +
            weights['survival'] * survival +
            weights['leverage_penalty'] * leverage_penalty +
            weights['consistency_bonus'] * consistency_bonus
        ) * timeframe_factor
        
        return float(np.clip(total_reward * self.reward_scaling, self.min_reward, self.max_reward))
    def _get_timeframe_factor(self) -> float:
        """Dynamic timeframe factor based on timeframe duration in minutes"""
        tf_minutes = min(self.timeframes)  # Use the most granular timeframe
        if tf_minutes < 5:    # 1m
            return 1.0
        elif tf_minutes < 15: # 5m
            return 1.2
        elif tf_minutes < 60: # 15m, 30m
            return 1.5
        elif tf_minutes < 240: # 1h, 2h
            return 1.8
        elif tf_minutes < 1440: # 4h, daily
            return 2.0
        else: # Weekly, monthly
            return 2.5

    def _calculate_sl_tp_levels(self, symbol: str, order_type: OrderType) -> Tuple[float, float]:
        """Calculate realistic SL/TP levels based on market volatility"""
        df = self.simulator.symbols_data[(symbol, min(self.timeframes))]
        nearest = self.nearest_time(symbol, self.simulator.current_time)
        current_idx = df.index.get_loc(nearest)
        
        lookback = min(14, current_idx)
        highs = df.iloc[current_idx-lookback:current_idx+1]['High'].values
        lows = df.iloc[current_idx-lookback:current_idx+1]['Low'].values
        atr = np.mean(highs - lows)
        
        current_price = df.iloc[current_idx]['Close']
        
        if order_type == OrderType.Buy:
            sl = current_price - self.atr_multiplier * atr
            tp = current_price + self.min_tp_sl_ratio * self.atr_multiplier * atr
        else:
            sl = current_price + self.atr_multiplier * atr
            tp = current_price - self.min_tp_sl_ratio * self.atr_multiplier * atr
        
        return sl, tp

    def _update_position_times(self):
        current_time = self.time_points[self._current_tick]
        active_ids = {order.id for order in self.simulator.orders}
        
        for order_id in list(self.position_start_times.keys()):
            if order_id not in active_ids:
                holding_time = (current_time - self.position_start_times.pop(order_id)).total_seconds() / 3600  # in hours
                for trade in self.trade_history:
                    if trade.get('order_id') == order_id:
                        trade['holding_time'] = holding_time
        
        for order in self.simulator.orders:
            if order.id not in self.position_start_times:
                self.position_start_times[order.id] = current_time

    def _update_drawdown_metrics(self):
        current_equity = self.simulator.equity
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity) / (self.peak_equity + 1e-6)
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def _calculate_win_rate(self) -> float:
        if not self.trade_history:
            return 0.0
        wins = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        return wins / len(self.trade_history)

    def _calculate_sharpe_ratio(self, annualize_factor=252) -> float:
        if len(self.trade_history) < 2:
            return 0.0
        returns = np.array([trade['profit'] for trade in self.trade_history])
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return mean_return / (std_return + 1e-6) * np.sqrt(annualize_factor)

    def _calculate_sortino_ratio(self, annualize_factor=252) -> float:
        if len(self.trade_history) < 2:
            return 0.0
        returns = np.array([trade['profit'] for trade in self.trade_history])
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.mean(returns) * annualize_factor / (np.std(returns) + 1e-6)
        downside_std = np.std(downside_returns)
        return np.mean(returns) * np.sqrt(annualize_factor) / (downside_std + 1e-6)

    def _calculate_profit_factor(self) -> float:
        if not self.trade_history:
            return 0.0
        gross_profit = sum(trade['profit'] for trade in self.trade_history if trade['profit'] > 0)
        gross_loss = abs(sum(trade['profit'] for trade in self.trade_history if trade['profit'] < 0))
        return gross_profit / (gross_loss + 1e-6)

    

    def _get_prices(self) -> Dict[str, np.ndarray]:
        prices = {}
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.simulator.symbols_data[(symbol, tf)]
                ohlc = []
                for col in ['Open', 'High', 'Low', 'Close']:
                    arr = df[col].reindex(self.time_points, method='ffill').values.reshape(-1, 1)
                    ohlc.append(arr)
                prices[(symbol, tf)] = np.concatenate(ohlc, axis=1)
        return prices

    def _process_data(self) -> np.ndarray:
        """
        Optimized feature extraction: ~100 core features, then PCA to 10 components.
        Loads from cache if available and valid, otherwise computes and caches.
        """
        import os
        from tqdm import tqdm
        cache_path = self.optimized_feature_engine.cache_path

        data = self.prices
        dt_index = self.simulator.symbols_data[(self.symbols[0], self.timeframes[0])].reindex(self.time_points, method='ffill').index

        ohlc_features = np.column_stack([
            data[(symbol, tf)] for symbol in self.symbols for tf in self.timeframes
        ])
        range_features = np.column_stack([
            (data[(symbol, tf)][:, 1] - data[(symbol, tf)][:, 2]).reshape(-1, 1)
            for symbol in self.symbols for tf in self.timeframes
        ])
        price_action_feats = np.concatenate([ohlc_features, range_features], axis=1)

        sim = self.simulator
        liquidity_feats = []
        for idx, dt in enumerate(tqdm(dt_index, desc='Liquidity Features')):
            row = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    dist_liq, liq_sweep, liq_buildup, ob_strength = sim.get_liquidity_features(symbol, tf, dt)
                    row.extend([dist_liq, liq_sweep, liq_buildup, ob_strength])
            liquidity_feats.append(row)
        liquidity_feats = np.array(liquidity_feats)

        hour = np.array([d.hour for d in dt_index])
        minute = np.array([d.minute for d in dt_index])
        weekday = np.array([d.weekday() for d in dt_index])
        hour_sin = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
        hour_cos = np.cos(2 * np.pi * hour / 24).reshape(-1, 1)
        minute_sin = np.sin(2 * np.pi * minute / 60).reshape(-1, 1)
        minute_cos = np.cos(2 * np.pi * minute / 60).reshape(-1, 1)
        weekday_sin = np.sin(2 * np.pi * weekday / 7).reshape(-1, 1)
        weekday_cos = np.cos(2 * np.pi * weekday / 7).reshape(-1, 1)
        time_arr = np.linspace(0, 1, len(self.time_points)).reshape(-1, 1)
        temporal_feats = np.concatenate([
            time_arr, hour_sin, hour_cos, minute_sin, minute_cos, weekday_sin, weekday_cos
        ], axis=1)

        structure_feats = []
        for idx, dt in enumerate(tqdm(dt_index, desc='Market Structure')):
            row = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    bos_choch = sim.get_bos_choch_flags(symbol, tf)
                    trend = sim.get_trend_state(symbol, tf)
                    weak_high, strong_high, weak_low, strong_low = sim.get_weak_strong_high_low(symbol, tf)
                    eqh, eql = sim.get_equal_high_low_count(symbol, tf)
                    row.extend([
                        *bos_choch.flatten(), trend,
                        int(weak_high), int(strong_high), int(weak_low), int(strong_low),
                        eqh, eql
                    ])
            structure_feats.append(row)
        structure_feats = np.array(structure_feats)

        risk_feats = []
        for idx, dt in enumerate(tqdm(dt_index, desc='Risk Features')):
            row = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    rr_pot, stop_hunt, atr15, london_bias, ny_bias = sim.get_trade_risk_features(symbol, tf, dt)
                    row.extend([rr_pot, stop_hunt, atr15, london_bias, ny_bias])
            risk_feats.append(row)
        risk_feats = np.array(risk_feats)

        core_features = np.concatenate([
            price_action_feats, liquidity_feats, temporal_feats, structure_feats, risk_feats
        ], axis=1)

        features_reduced = self.optimized_feature_engine.fit_transform(core_features)

        return features_reduced

    def _get_observation(self) -> Dict[str, np.ndarray]:
        features = self.signal_features[(self._current_tick-self.window_size+1):(self._current_tick+1)]

        orders = np.zeros(self.observation_space['orders'].shape)
        for i, symbol in enumerate(self.symbols):
            symbol_orders = self.simulator.symbol_orders(symbol)
            for j, order in enumerate(symbol_orders):
                orders[i, j] = [order.entry_price, order.volume, order.profit]

        return {
            'balance': np.array([self.simulator.balance]),
            'equity': np.array([self.simulator.equity]),
            'margin': np.array([self.simulator.margin]),
            'features': features,
            'orders': orders,
        }

    def _create_info(self, **kwargs: Any) -> Dict[str, Any]:
        info = {
            'balance': self.simulator.balance,
            'equity': self.simulator.equity,
            'margin': self.simulator.margin,
            'free_margin': self.simulator.free_margin,
            'margin_level': self.simulator.margin_level,
            'current_time': self.time_points[self._current_tick],
            'num_positions': len(self.simulator.orders),
            'peak_equity': self.peak_equity,
            'max_drawdown': self.max_drawdown,
            'win_rate': self._calculate_win_rate(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'profit_factor': self._calculate_profit_factor(),
            'avg_trade_profit': np.mean([t['profit'] for t in self.trade_history]) if self.trade_history else 0,
            'avg_holding_time': np.mean([t.get('holding_time', 0) for t in self.trade_history]) if self.trade_history else 0,
            'volatility': self.returns_volatility,
            'total_trades': len(self.trade_history),
            'portfolio_value': self.simulator.equity
        }
        info.update(kwargs)
        return info

    def _apply_action(self, action: np.ndarray) -> Tuple[Dict, Dict]:
        action = np.clip(action, -1.0, 1.0)
        
        orders_info = {}
        closed_orders_info = {symbol: [] for symbol in self.symbols}
        k = self.symbol_max_orders + 2

        for i, symbol in enumerate(self.symbols):
            symbol_action = action[k*i:k*(i+1)]
            close_orders_logit = symbol_action[:-2] * 5
            hold_logit = symbol_action[-2] * 5
            volume = symbol_action[-1]

            close_orders_probability = expit(close_orders_logit)
            hold_probability = expit(hold_logit)
            hold = bool(hold_probability > self.hold_threshold)
            modified_volume = self._get_modified_volume(symbol, volume)

            symbol_orders = self.simulator.symbol_orders(symbol)
            orders_to_close_index = np.where(
                close_orders_probability[:len(symbol_orders)] > self.close_threshold
            )[0]
            orders_to_close = np.array(symbol_orders)[orders_to_close_index]

            for j, order in enumerate(orders_to_close):
                profit = self.simulator.close_order(order)
                closed_orders_info[symbol].append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'order_type': order.type,
                    'volume': order.volume,
                    'fee': order.fee,
                    'margin': order.margin,
                    'profit': profit,
                    'close_probability': close_orders_probability[orders_to_close_index][j],
                })

            orders_capacity = self.symbol_max_orders - (len(symbol_orders) - len(orders_to_close))
            orders_info[symbol] = {
                'order_id': None,
                'symbol': symbol,
                'hold_probability': hold_probability,
                'hold': hold,
                'volume': volume,
                'capacity': orders_capacity,
                'order_type': None,
                'modified_volume': modified_volume,
                'fee': float('nan'),
                'margin': float('nan'),
                'error': '',
            }

            if not hold and orders_capacity > 0:
                order_type = OrderType.Buy if volume > 0 else OrderType.Sell
                fee = self.fee if isinstance(self.fee, float) else self.fee(symbol)

                try:
                    order = self.simulator.create_order(order_type, symbol, modified_volume, fee)
                    orders_info[symbol].update({
                        'order_id': order.id,
                        'order_type': order_type,
                        'fee': fee,
                        'margin': order.margin,
                    })
                except ValueError as e:
                    orders_info[symbol]['error'] = str(e)

        return orders_info, closed_orders_info

    def _get_modified_volume(self, symbol: str, volume: float) -> float:
        si = self.simulator.symbols_info[symbol]
        v = abs(volume)
        v = np.clip(v, si.volume_min, si.volume_max)
        v = round(v / si.volume_step) * si.volume_step
        return v

    def render(self, mode: str = 'human', **kwargs: Any) -> Any:
        if mode == 'simple_figure':
            return self._render_simple_figure(**kwargs)
        if mode == 'advanced_figure':
            return self._render_advanced_figure(**kwargs)
        return self.simulator.get_state(**kwargs)

    def _render_simple_figure(self, figsize=(14, 6), return_figure=False):
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.symbols)))

        for j, symbol in enumerate(self.symbols):
            close_price = self.prices[(symbol, 1)][:, 0]
            symbol_color = symbol_colors[j]

            ax.plot(self.time_points, close_price, c=symbol_color, marker='.', label=symbol)

            buy_ticks, buy_error_ticks = [], []
            sell_ticks, sell_error_ticks = [], []
            close_ticks = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1
                order = self.history[i]['orders'].get(symbol, {})
                
                if order and not order['hold']:
                    if order['order_type'] == OrderType.Buy:
                        if order['error']:
                            buy_error_ticks.append(tick)
                        else:
                            buy_ticks.append(tick)
                    else:
                        if order['error']:
                            sell_error_ticks.append(tick)
                        else:
                            sell_ticks.append(tick)

                if self.history[i]['closed_orders'].get(symbol):
                    close_ticks.append(tick)

            tp = np.array(self.time_points)
            ax.plot(tp[buy_ticks], close_price[buy_ticks], '^', color='green')
            ax.plot(tp[buy_error_ticks], close_price[buy_error_ticks], '^', color='gray')
            ax.plot(tp[sell_ticks], close_price[sell_ticks], 'v', color='red')
            ax.plot(tp[sell_error_ticks], close_price[sell_error_ticks], 'v', color='gray')
            ax.plot(tp[close_ticks], close_price[close_ticks], '|', color='black')

            ax.tick_params(axis='y', labelcolor=symbol_color)
            ax.yaxis.tick_left()
            if j < len(self.symbols) - 1:
                ax = ax.twinx()

        fig.suptitle(
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} | "
            f"Equity: {self.simulator.equity:.6f} | "
            f"Margin: {self.simulator.margin:.6f} | "
            f"Free Margin: {self.simulator.free_margin:.6f} | "
            f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.legend(loc='right')

        return fig if return_figure else plt.show()

    def _render_advanced_figure(self, figsize=(1400, 600), time_format="%Y-%m-%d %H:%m", return_figure=False):
        fig = go.Figure()
        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.symbols)))
        get_color_string = lambda color: f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})"

        extra_info = [
            f"Time: {self.time_points[i]}<br>"
            f"Balance: {h['balance']:.6f} {self.simulator.unit}<br>"
            f"Equity: {h['equity']:.6f}<br>"
            f"Margin: {h['margin']:.6f}<br>"
            f"Free Margin: {h['free_margin']:.6f}<br>"
            f"Margin Level: {h['margin_level']:.6f}"
            for i, h in enumerate(self.history)
        ]
        extra_info = [extra_info[0]] * (self.window_size - 1) + extra_info

        for j, symbol in enumerate(self.symbols):
            close_price = self.prices[(symbol, 1)][:, 0]
            symbol_color = symbol_colors[j]

            fig.add_trace(go.Scatter(
                x=self.time_points,
                y=close_price,
                mode='lines+markers',
                line_color=get_color_string(symbol_color),
                opacity=1.0,
                hovertext=extra_info,
                name=symbol,
                yaxis=f'y{j+1}',
                legendgroup=f'g{j+1}',
            ))

            fig.update_layout(**{
                f'yaxis{j+1}': dict(
                    tickfont=dict(color=get_color_string(symbol_color * [1, 1, 1, 0.8])),
                    overlaying='y' if j > 0 else None,
                ),
            })

            trade_ticks, trade_markers = [], []
            trade_colors, trade_sizes = [], []
            trade_extra_info = []
            trade_max_volume = max([
                h.get('orders', {}).get(symbol, {}).get('modified_volume') or 0
                for h in self.history
            ])
            close_ticks, close_extra_info = [], []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1
                order = self.history[i]['orders'].get(symbol)
                
                if order and not order['hold']:
                    size = 8 + 22 * (order['modified_volume'] / (trade_max_volume + 1e-6))
                    info = (
                        f"Order ID: {order['order_id'] or ''}<br>"
                        f"Type: {'Buy' if order['volume'] > 0 else 'Sell'}<br>"
                        f"Volume: {order['modified_volume']:.4f}<br>"
                        f"Hold Prob: {order['hold_probability']:.2f}<br>"
                        f"Error: {order['error'] or 'None'}"
                    )
                    
                    if order['order_type'] == OrderType.Buy:
                        marker = 'triangle-up'
                        color = 'gray' if order['error'] else 'green'
                    else:
                        marker = 'triangle-down'
                        color = 'gray' if order['error'] else 'red'

                    trade_ticks.append(tick)
                    trade_markers.append(marker)
                    trade_colors.append(color)
                    trade_sizes.append(size)
                    trade_extra_info.append(info)

                if self.history[i]['closed_orders'].get(symbol):
                    info = "<br>".join([
                        f"Closed Order {o['order_id']}<br>"
                        f"Type: {o['order_type'].name}<br>"
                        f"Profit: {o['profit']:.4f}"
                        for o in self.history[i]['closed_orders'][symbol]
                    ])
                    close_ticks.append(tick)
                    close_extra_info.append(info)

            fig.add_trace(go.Scatter(
                x=np.array(self.time_points)[trade_ticks],
                y=close_price[trade_ticks],
                mode='markers',
                hovertext=trade_extra_info,
                marker_symbol=trade_markers,
                marker_color=trade_colors,
                marker_size=trade_sizes,
                name=symbol,
                yaxis=f'y{j+1}',
                showlegend=False,
                legendgroup=f'g{j+1}',
            ))

            fig.add_trace(go.Scatter(
                x=np.array(self.time_points)[close_ticks],
                y=close_price[close_ticks],
                mode='markers',
                hovertext=close_extra_info,
                marker_symbol='line-ns',
                marker_color='black',
                marker_size=7,
                marker_line_width=1.5,
                name=symbol,
                yaxis=f'y{j+1}',
                showlegend=False,
                legendgroup=f'g{j+1}',
            ))

        fig.update_layout(
            title=(
                f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} | "
                f"Equity: {self.simulator.equity:.6f} | "
                f"Margin: {self.simulator.margin:.6f}"
            ),
            xaxis_tickformat=time_format,
            width=figsize[0],
            height=figsize[1],
        )

        return fig if return_figure else fig.show()

    def close(self) -> None:
        plt.close()
        if self.multiprocessing_pool:
            self.multiprocessing_pool.close()
