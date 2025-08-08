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
        atr_multiplier: float = 2.0,     # ATR multiplier for SL
        use_cached_features: bool = False,  # Prefer cached features when available
        execution_cost_penalty_weight: float = 0.10,
        max_daily_drawdown: float = 0.20,
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
        self.use_cached_features = use_cached_features
        self.execution_cost_penalty_weight = execution_cost_penalty_weight
        self.max_daily_drawdown = max_daily_drawdown

        self.multiprocessing_pool = Pool(multiprocessing_processes) if multiprocessing_processes else None
        self.render_mode = render_mode
        
        self.simulator = MtSimulator(
            symbols=symbols,
            timeframes=timeframes,
            start=time_start,
            end=time_end,
            enable_realistic_execution=True  # Enable realistic market simulation
        )
        
        self.time_points = self.simulator.symbols_data[(symbols[0], timeframes[0])].index.to_pydatetime().tolist()
        self.simulator.current_time = self.time_points[0]
        self.prices = self._get_prices()
        
        self.optimized_feature_engine = OptimizedFeatureEngine(
            target_variance=0.95,
            min_components=20,
            max_components=50,
            variance_threshold=1e-6,
            cache_path=f"feature_cache_{'_'.join(self.symbols)}_{'_'.join(map(str, self.timeframes))}.pkl",
            reuse_existing=True,
            keep_last=2,
        )

        if self.use_cached_features:
            try:
                self.signal_features = self.optimized_feature_engine.load_cache()
                print(f"✅ Loaded cached features with shape {self.signal_features.shape}")
            except Exception as e:
                raise RuntimeError(
                    "Cached features not found. Please run a pre-cache step in the main process before launching workers."
                )
            self.liquidity_features = self._compute_liquidity_only()
        else:
            self.signal_features = self._process_data()
            if not hasattr(self, 'liquidity_features') or self.liquidity_features is None:
                self.liquidity_features = self._compute_liquidity_only()
        self.features_shape = (window_size, self.signal_features.shape[1])
        self.liquidity_shape = (window_size, self.liquidity_features.shape[1])
        
        INF = 1e10
        self._action_dims_per_symbol = self.symbol_max_orders + 5
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float64,
            shape=(len(self.symbols) * self._action_dims_per_symbol,))
        
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'equity': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'margin': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'features': spaces.Box(low=-INF, high=INF, shape=self.features_shape, dtype=np.float64),
            'liquidity': spaces.Box(low=-INF, high=INF, shape=self.liquidity_shape, dtype=np.float64),
            'orders': spaces.Box(
                low=-INF, high=INF, dtype=np.float64,
                shape=(len(self.symbols) * self.symbol_max_orders * 3,)  # Flatten to 1D to avoid vision network
            )
        })
        
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.time_points) - 1
        self._truncated = False
        self._current_tick = self._start_tick
        self.history = []
        
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

    def reset(self, seed=None, options=None):
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

        obs = self._get_observation()
        info = self._create_info()
        if not (isinstance(obs, dict) and all(isinstance(v, np.ndarray) for v in obs.values())):
            print("[ERROR] MtEnv.reset: Observation is not a dict of np.ndarray!", type(obs), {k: type(v) for k, v in obs.items()})
        return obs, info

    def step(self, action: np.ndarray):
        orders_info, closed_orders_info = self._apply_action(action)
        self._current_tick += 1
        terminated = self._current_tick == self._end_tick
        truncated = False  # You can set this to True if you have a time limit or other truncation condition
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
        self._update_position_times()
        self._update_drawdown_metrics()
        if self.max_drawdown > self.max_daily_drawdown:
            terminated = True
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
        if not (isinstance(observation, dict) and all(isinstance(v, np.ndarray) for v in observation.values())):
            print("[ERROR] MtEnv.step: Observation is not a dict of np.ndarray!", type(observation), {k: type(v) for k, v in observation.items()})
        return observation, step_reward, terminated, truncated, info

    def _calculate_reward(self, orders_info: Dict, closed_orders_info: Dict) -> float:
        """Enhanced reward calculation with volatility normalization and progressive penalties"""
        prev_info = self.history[-1]
        current_equity = self.simulator.equity
        
        equity_change = (current_equity - prev_info['equity']) / (prev_info['equity'] + 1e-6)
        
        if len(self.history) >= 20:
            recent_equity = [h['equity'] for h in self.history[-20:]]
            equity_returns = np.diff(recent_equity) / (np.array(recent_equity[:-1]) + 1e-6)
            volatility = np.std(equity_returns) + 1e-6
            normalized_equity_change = equity_change / volatility
        else:
            normalized_equity_change = equity_change
            
        self.episode_reward_components['equity_change'].append(normalized_equity_change)
        
        exec_entries = [e for e in getattr(self.simulator, 'execution_history', []) if e.get('time') == prev_info.get('current_time')]
        is_penalty = 0.0
        adverse_penalty = 0.0
        participation_penalty = 0.0
        queue_loss_penalty = 0.0
        total_cost_bps = 0.0
        for e in exec_entries:
            market_px = float(e.get('market_price', 0.0))
            exec_px = float(e.get('execution_price', market_px))
            side = 1.0 if e.get('order_type') == 'Buy' else -1.0
            is_bps = ((exec_px - market_px) / max(market_px, 1e-9)) * 10000.0 * side
            is_penalty -= is_bps / 10000.0  # negative impact reduces reward
            total_cost_bps += abs(float(e.get('slippage_bps', 0.0))) + float(e.get('spread_bps', 0.0))
            adverse_penalty -= float(e.get('adverse_selection_bps', 0.0)) / 10000.0
            pr = float(e.get('participation_rate', 0.0))
            depth = float(e.get('book_total_depth', 1.0))
            participation_penalty -= (pr ** 1.2) / (np.log1p(depth) + 1e-6)
            fill_ratio = float(e.get('fill_ratio', 1.0))
            queue_loss_penalty -= (1.0 - fill_ratio) * 0.01
        execution_penalty = - self.execution_cost_penalty_weight * (total_cost_bps / 10000.0)
        self.episode_reward_components['execution_cost_bps'].append(total_cost_bps)
        self.episode_reward_components['is_penalty'].append(is_penalty)
        self.episode_reward_components['adverse_penalty'].append(adverse_penalty)
        self.episode_reward_components['participation_penalty'].append(participation_penalty)
        self.episode_reward_components['queue_loss_penalty'].append(queue_loss_penalty)
        
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
                        annualized_roi = 0
                else:
                    annualized_roi = 0
                if len(self.trade_history) > 5:
                    recent_returns = np.array([t['profit']/t['margin'] for t in self.trade_history[-10:] if 'margin' in t])
                    if len(recent_returns) > 1:
                        sharpe_like = roi / (np.std(recent_returns) + 1e-6)
                        trade_quality += 0.6 * np.sign(roi) * np.log1p(abs(sharpe_like))
                trade_quality += 0.4 * np.sign(roi) * np.log1p(abs(annualized_roi))
                if roi > 0.01 and 'holding_time' in order:
                    efficiency_bonus = self.early_close_bonus * min(1.0, roi / (order['holding_time'] + 1e-6))
                    trade_quality += efficiency_bonus
        self.episode_reward_components['trade_quality'].append(trade_quality)
        
        margin_level = self.simulator.margin_level
        if margin_level < 5.0:
            risk_penalty = -1.0 * (5.0 - margin_level) ** 2
        elif margin_level < 10.0:
            risk_penalty = -0.3 * (10.0 - margin_level) ** 1.5
        else:
            risk_penalty = 0
        if self.max_drawdown > 0.15:
            drawdown_penalty = -2.0 * (self.max_drawdown - 0.15) ** 2
        elif self.max_drawdown > 0.10:
            drawdown_penalty = -0.5 * (self.max_drawdown - 0.10) ** 2
        elif self.max_drawdown > 0.05:
            drawdown_penalty = -0.1 * (self.max_drawdown - 0.05) ** 2
        else:
            drawdown_penalty = 0
        tail_penalty = 0.0
        if len(self.portfolio_values) > 30:
            rets = np.diff(self.portfolio_values) / (np.array(self.portfolio_values[:-1]) + 1e-6)
            var_p = np.percentile(rets, 5)
            cvar = np.mean(rets[rets <= var_p]) if np.any(rets <= var_p) else 0.0
            tail_penalty = -abs(cvar) * 10.0
        total_risk_penalty = risk_penalty + self.drawdown_penalty * drawdown_penalty + tail_penalty
        self.episode_reward_components['risk_penalty'].append(total_risk_penalty)
        
        open_positions = len(self.simulator.orders)
        position_bonus = 0.05 * min(open_positions, len(self.symbols))
        unique_symbols = len({order.symbol for order in self.simulator.orders})
        diversification = self.diversification_bonus * unique_symbols / len(self.symbols)
        
        if len(self.history) >= 10:
            recent_prices = [h.get('close_price', h.get('equity', 0)) for h in self.history[-10:]]
            market_volatility = np.std(np.diff(recent_prices)) / (np.mean(recent_prices) + 1e-6)
            if market_volatility < 0.005:
                regime_factor = 0.8
            elif market_volatility > 0.02:
                regime_factor = 1.2
            else:
                regime_factor = 1.0
        else:
            regime_factor = 1.0
        
        weights = {
            'equity_change': 0.25,
            'trade_quality': 0.25,
            'execution': 0.15,
            'is': 0.10,
            'adverse': 0.05,
            'participation': 0.05,
            'queue': 0.05,
            'risk': 0.10
        }
        total_reward = (
            weights['equity_change'] * normalized_equity_change +
            weights['trade_quality'] * trade_quality * self.trade_reward_multiplier +
            weights['execution'] * execution_penalty +
            weights['is'] * is_penalty +
            weights['adverse'] * adverse_penalty +
            weights['participation'] * participation_penalty +
            weights['queue'] * queue_loss_penalty +
            weights['risk'] * total_risk_penalty
        ) * regime_factor
        
        survival = self.survival_bonus if not self._truncated else 0
        leverage = self.simulator.margin / (self.simulator.balance + 1e-6)
        leverage_penalty = -self.leverage_penalty * (max(leverage - 1, 0) ** 2)
        total_reward += survival + leverage_penalty
        
        dynamic_max = self.max_reward * regime_factor
        dynamic_min = self.min_reward * regime_factor
        
        return float(np.clip(total_reward * self.reward_scaling, dynamic_min, dynamic_max))
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
        Enhanced feature extraction with improved liquidity dynamics and regime detection.
        Includes target-aware feature selection for better explained variance.
        """
        import os
        from tqdm import tqdm
        data = self.prices
        dt_index = self.simulator.symbols_data[(self.symbols[0], self.timeframes[0])].reindex(self.time_points, method='ffill').index

        ohlc_features = []
        for symbol in self.symbols:
            for tf in self.timeframes:
                ohlc = data[(symbol, tf)]
                basic_ohlc = ohlc.copy()
                close_prices = ohlc[:, 0]
                high_low_ratio = (ohlc[:, 1] - ohlc[:, 2]) / (close_prices + 1e-8)
                open_close_ratio = (ohlc[:, 0] - ohlc[:, 3]) / (close_prices + 1e-8)
                returns = np.diff(np.log(close_prices + 1e-8))
                volatility = np.zeros_like(close_prices)
                volatility[1:] = np.abs(returns)
                rolling_vol = np.zeros_like(close_prices)
                for i in range(20, len(close_prices)):
                    rolling_vol[i] = np.std(returns[max(0, i-20):i])
                combined = np.column_stack([
                    basic_ohlc,
                    high_low_ratio.reshape(-1, 1),
                    open_close_ratio.reshape(-1, 1),
                    volatility.reshape(-1, 1),
                    rolling_vol.reshape(-1, 1)
                ])
                ohlc_features.append(combined)
        ohlc_features = np.concatenate(ohlc_features, axis=1)

        sim = self.simulator
        enhanced_liquidity_feats = []
        vpin_state: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for symbol in self.symbols:
            for tf in self.timeframes:
                vpin_state[(symbol, tf)] = {
                    'prev_mid': None,
                    'buy_vol': 0.0,
                    'sell_vol': 0.0,
                    'vpin_window': deque(maxlen=50),
                }
        for idx, dt in enumerate(tqdm(dt_index, desc='Enhanced Liquidity Features')):
            row = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    df = sim.symbols_data[(symbol, tf)]
                    nearest = sim.nearest_time(symbol, dt, tf)
                    current_idx = df.index.get_loc(nearest)
                    if current_idx >= 10:
                        recent_data = df.iloc[max(0, current_idx-10):current_idx+1]
                    else:
                        recent_data = df.iloc[:current_idx+1]
                    recent_prices = recent_data['Close'].values
                    current_price = float(recent_prices[-1]) if len(recent_prices) else float(df.iloc[current_idx]['Close'])

                    if hasattr(sim, 'market_impact_model') and sim.market_impact_model:
                        bid, ask = sim.market_impact_model.get_dynamic_spread(
                            symbol, dt, recent_prices, current_price
                        )
                        mid = (bid + ask) / 2.0
                        spread_bps = ((ask - bid) / max(current_price, 1e-9)) * 10000.0
                        lob_metrics = sim.market_impact_model.get_lob_metrics(
                            symbol, dt, recent_prices, current_price
                        )
                        total_depth = lob_metrics['total_depth']
                        imbalance = lob_metrics['imbalance']
                    else:
                        mid = current_price
                        spread_bps = 0.0
                        total_depth = 1.0
                        imbalance = 0.0

                    vol_col = None
                    for c in ['TickVolume', 'Tick_Volume', 'Volume', 'Real_Volume']:
                        if c in df.columns:
                            vol_col = c
                            break
                    volume_proxy = float(df.loc[nearest][vol_col]) if vol_col else max(1.0, total_depth)

                    effective_spread_bps = 2.0 * abs(current_price - mid) / max(mid, 1e-9) * 10000.0

                    if current_idx + 3 < len(df):
                        future_close = float(df.iloc[current_idx + 3]['Close'])
                        realized_spread_bps = 2.0 * (current_price - future_close) / max(mid, 1e-9) * 10000.0
                    else:
                        realized_spread_bps = 0.0

                    prev_mid = vpin_state[(symbol, tf)]['prev_mid']
                    if prev_mid is None:
                        delta_mid = 0.0
                    else:
                        delta_mid = mid - prev_mid
                    ofi_proxy = (delta_mid / max(prev_mid if prev_mid else mid, 1e-9)) * total_depth if prev_mid else 0.0
                    footprint_delta = np.sign(delta_mid) * volume_proxy
                    vpin_state[(symbol, tf)]['prev_mid'] = mid

                    if delta_mid >= 0:
                        vpin_state[(symbol, tf)]['buy_vol'] += volume_proxy
                    else:
                        vpin_state[(symbol, tf)]['sell_vol'] += volume_proxy
                    b = vpin_state[(symbol, tf)]['buy_vol']
                    s = vpin_state[(symbol, tf)]['sell_vol']
                    vpin_inst = abs(b - s) / max(b + s, 1e-6)
                    vpin_state[(symbol, tf)]['vpin_window'].append(vpin_inst)
                    vpin_proxy = float(np.mean(vpin_state[(symbol, tf)]['vpin_window']))

                    trade_count = float(df.loc[nearest][vol_col]) if vol_col else max(1.0, total_depth)
                    bar_seconds = int(tf) * 60
                    inter_trade_time = bar_seconds / max(trade_count, 1e-6)

                    dist_liq, liq_sweep, liq_buildup, ob_strength = sim.get_liquidity_features(symbol, tf, dt)
                    sessions, mins_since_open, magic1, magic2 = sim.get_session_info(dt)
                    hourly_liquidity = sim.market_impact_model.get_daily_liquidity_pattern(dt.hour) if hasattr(sim, 'market_impact_model') and sim.market_impact_model else 1.0

                    if current_idx >= 10:
                        market_depth_proxy = 1.0 / (np.std(recent_prices) + 1e-8)
                    else:
                        market_depth_proxy = 1.0

                    row.extend([
                        spread_bps, effective_spread_bps, realized_spread_bps,
                        total_depth, imbalance, ofi_proxy, vpin_proxy, footprint_delta,
                        trade_count, inter_trade_time,
                        dist_liq, liq_sweep, liq_buildup, ob_strength,
                        *sessions, mins_since_open / 1440.0,
                        magic1 / 144.0, magic2 / 89.0,
                        hourly_liquidity, market_depth_proxy
                    ])
            enhanced_liquidity_feats.append(row)
        enhanced_liquidity_feats = np.array(enhanced_liquidity_feats)
        self.liquidity_features = enhanced_liquidity_feats

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
        is_london = ((hour >= 8) & (hour < 17)).astype(float).reshape(-1, 1)
        is_ny = ((hour >= 13) & (hour < 22)).astype(float).reshape(-1, 1)
        is_overlap = ((hour >= 13) & (hour < 17)).astype(float).reshape(-1, 1)
        is_weekend = (weekday >= 5).astype(float).reshape(-1, 1)
        temporal_feats = np.concatenate([
            time_arr, hour_sin, hour_cos, minute_sin, minute_cos,
            weekday_sin, weekday_cos, is_london, is_ny, is_overlap, is_weekend
        ], axis=1)

        structure_feats = []
        for idx, dt in enumerate(tqdm(dt_index, desc='Enhanced Market Structure')):
            row = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    bos_choch = sim.get_bos_choch_flags(symbol, tf)
                    trend = sim.get_trend_state(symbol, tf)
                    weak_high, strong_high, weak_low, strong_low = sim.get_weak_strong_high_low(symbol, tf)
                    eqh, eql = sim.get_equal_high_low_count(symbol, tf)
                    cycle_state = sim.get_cycle_state(symbol, tf, dt)
                    df = sim.symbols_data[(symbol, tf)]
                    nearest = sim.nearest_time(symbol, dt, tf)
                    current_idx = df.index.get_loc(nearest)
                    if current_idx >= 20:
                        recent_closes = df.iloc[max(0, current_idx-20):current_idx+1]['Close'].values
                        trend_strength = abs(np.corrcoef(np.arange(len(recent_closes)), recent_closes)[0, 1])
                        if np.isnan(trend_strength):
                            trend_strength = 0.0
                    else:
                        trend_strength = 0.0
                    row.extend([
                        *bos_choch.flatten(), trend, trend_strength,
                        int(weak_high), int(strong_high), int(weak_low), int(strong_low),
                        eqh, eql, cycle_state
                    ])
            structure_feats.append(row)
        structure_feats = np.array(structure_feats)

        risk_feats = []
        for idx, dt in enumerate(tqdm(dt_index, desc='Enhanced Risk Features')):
            row = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    rr_pot, stop_hunt, atr15, london_bias, ny_bias = sim.get_trade_risk_features(symbol, tf, dt)
                    df = sim.symbols_data[(symbol, tf)]
                    nearest = sim.nearest_time(symbol, dt, tf)
                    current_idx = df.index.get_loc(nearest)
                    current_price = df.iloc[current_idx]['Close']
                    if current_idx >= 24:
                        recent_prices = df.iloc[max(0, current_idx-24):current_idx+1]['Close'].values
                        vol_regime = np.std(np.diff(np.log(recent_prices)))
                    else:
                        vol_regime = 0.01
                    session_multiplier = 1.0
                    if dt.hour >= 8 and dt.hour < 17:
                        session_multiplier = 0.7
                    elif dt.hour >= 13 and dt.hour < 17:
                        session_multiplier = 0.5
                    elif dt.hour >= 17 and dt.hour < 22:
                        session_multiplier = 0.8
                    else:
                        session_multiplier = 1.5
                    estimated_spread = vol_regime * session_multiplier * 10000
                    row.extend([
                        rr_pot, stop_hunt, atr15, london_bias, ny_bias,
                        vol_regime, estimated_spread
                    ])
            risk_feats.append(row)
        risk_feats = np.array(risk_feats)

        core_features = np.concatenate([
            ohlc_features, enhanced_liquidity_feats, temporal_feats,
            structure_feats, risk_feats
        ], axis=1)
        print(f"📊 Core features shape: {core_features.shape}")

        target = None
        if len(core_features) > 1:
            close_prices = data[(self.symbols[0], self.timeframes[0])][:, 0]
            future_returns = np.zeros(len(close_prices))
            for i in range(len(close_prices) - 5):
                future_returns[i] = (close_prices[i + 5] - close_prices[i]) / close_prices[i]
            target = np.digitize(future_returns, bins=np.percentile(future_returns, [25, 75]))

        features_reduced = self.optimized_feature_engine.fit_transform(core_features, target)
        if self.use_cached_features and features_reduced is not None and not isinstance(features_reduced, dict):
            print(f"✅ Loaded cached features with shape {features_reduced.shape}")
        importance_info = self.optimized_feature_engine.get_feature_importance()
        print(f"📈 Feature Engineering Summary:")
        print(f"   - Input features: {core_features.shape[1]}")
        print(f"   - Output features: {features_reduced.shape[1]}")
        print(f"   - Explained variance: {importance_info.get('total_explained_variance', 'N/A'):.3f}")
        print(f"   - PCA components: {importance_info.get('n_components', 'N/A')}")

        return features_reduced

    def _compute_liquidity_only(self) -> np.ndarray:
        """Compute the liquidity-only feature block used as a separate observation branch."""
        sim = self.simulator
        dt_index = self.simulator.symbols_data[(self.symbols[0], self.timeframes[0])].reindex(self.time_points, method='ffill').index
        from collections import deque
        enhanced_liquidity_feats = []
        vpin_state: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for symbol in self.symbols:
            for tf in self.timeframes:
                vpin_state[(symbol, tf)] = {
                    'prev_mid': None,
                    'buy_vol': 0.0,
                    'sell_vol': 0.0,
                    'vpin_window': deque(maxlen=50),
                }
        for dt in dt_index:
            row = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    df = sim.symbols_data[(symbol, tf)]
                    nearest = sim.nearest_time(symbol, dt, tf)
                    current_idx = df.index.get_loc(nearest)
                    recent_data = df.iloc[max(0, current_idx-10):current_idx+1]
                    recent_prices = recent_data['Close'].values
                    current_price = float(recent_prices[-1]) if len(recent_prices) else float(df.iloc[current_idx]['Close'])
                    if hasattr(sim, 'market_impact_model') and sim.market_impact_model:
                        bid, ask = sim.market_impact_model.get_dynamic_spread(symbol, dt, recent_prices, current_price)
                        mid = (bid + ask) / 2.0
                        spread_bps = ((ask - bid) / max(current_price, 1e-9)) * 10000.0
                        lob_metrics = sim.market_impact_model.get_lob_metrics(symbol, dt, recent_prices, current_price)
                        total_depth = lob_metrics['total_depth']
                        imbalance = lob_metrics['imbalance']
                    else:
                        mid = current_price
                        spread_bps = 0.0
                        total_depth = 1.0
                        imbalance = 0.0
                    vol_col = None
                    for c in ['TickVolume', 'Tick_Volume', 'Volume', 'Real_Volume']:
                        if c in df.columns:
                            vol_col = c
                            break
                    volume_proxy = float(df.loc[nearest][vol_col]) if vol_col else max(1.0, total_depth)
                    effective_spread_bps = 2.0 * abs(current_price - mid) / max(mid, 1e-9) * 10000.0
                    if current_idx + 3 < len(df):
                        future_close = float(df.iloc[current_idx + 3]['Close'])
                        realized_spread_bps = 2.0 * (current_price - future_close) / max(mid, 1e-9) * 10000.0
                    else:
                        realized_spread_bps = 0.0
                    prev_mid = vpin_state[(symbol, tf)]['prev_mid']
                    delta_mid = 0.0 if prev_mid is None else (mid - prev_mid)
                    ofi_proxy = (delta_mid / max(prev_mid if prev_mid else mid, 1e-9)) * total_depth if prev_mid else 0.0
                    footprint_delta = np.sign(delta_mid) * volume_proxy
                    vpin_state[(symbol, tf)]['prev_mid'] = mid
                    if delta_mid >= 0:
                        vpin_state[(symbol, tf)]['buy_vol'] += volume_proxy
                    else:
                        vpin_state[(symbol, tf)]['sell_vol'] += volume_proxy
                    b = vpin_state[(symbol, tf)]['buy_vol']
                    s = vpin_state[(symbol, tf)]['sell_vol']
                    vpin_inst = abs(b - s) / max(b + s, 1e-6)
                    vpin_state[(symbol, tf)]['vpin_window'].append(vpin_inst)
                    vpin_proxy = float(np.mean(vpin_state[(symbol, tf)]['vpin_window']))
                    trade_count = float(df.loc[nearest][vol_col]) if vol_col else max(1.0, total_depth)
                    bar_seconds = int(tf) * 60
                    inter_trade_time = bar_seconds / max(trade_count, 1e-6)
                    dist_liq, liq_sweep, liq_buildup, ob_strength = sim.get_liquidity_features(symbol, tf, dt)
                    sessions, mins_since_open, magic1, magic2 = sim.get_session_info(dt)
                    hourly_liquidity = sim.market_impact_model.get_daily_liquidity_pattern(dt.hour) if hasattr(sim, 'market_impact_model') and sim.market_impact_model else 1.0
                    if len(recent_prices) >= 2:
                        market_depth_proxy = 1.0 / (np.std(recent_prices) + 1e-8)
                    else:
                        market_depth_proxy = 1.0
                    row.extend([
                        spread_bps, effective_spread_bps, realized_spread_bps,
                        total_depth, imbalance, ofi_proxy, vpin_proxy, footprint_delta,
                        trade_count, inter_trade_time,
                        dist_liq, liq_sweep, liq_buildup, ob_strength,
                        *sessions, mins_since_open / 1440.0,
                        magic1 / 144.0, magic2 / 89.0,
                        hourly_liquidity, market_depth_proxy
                    ])
            enhanced_liquidity_feats.append(row)
        return np.array(enhanced_liquidity_feats)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        features = self.signal_features[(self._current_tick-self.window_size+1):(self._current_tick+1)]
        liquidity = self.liquidity_features[(self._current_tick-self.window_size+1):(self._current_tick+1)]
        
        orders = np.zeros(self.observation_space['orders'].shape)
        flat_index = 0
        for i, symbol in enumerate(self.symbols):
            symbol_orders = self.simulator.symbol_orders(symbol)
            for j, order in enumerate(symbol_orders):
                if j < self.symbol_max_orders:  # Ensure we don't exceed the limit
                    orders[flat_index:flat_index+3] = [order.entry_price, order.volume, order.profit]
                    flat_index += 3

        return {
            'balance': np.array([self.simulator.balance]),
            'equity': np.array([self.simulator.equity]),
            'margin': np.array([self.simulator.margin]),
            'features': features,
            'liquidity': liquidity,
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
        k = self._action_dims_per_symbol

        for i, symbol in enumerate(self.symbols):
            symbol_action = action[k*i:k*(i+1)]
            close_orders_logit = symbol_action[:self.symbol_max_orders] * 5
            order_kind_logit = symbol_action[self.symbol_max_orders] * 5  # market vs limit
            tif_logit = symbol_action[self.symbol_max_orders + 1] * 5      # IOC vs FOK
            limit_offset_norm = symbol_action[self.symbol_max_orders + 2]  # [-1,1] -> [0, max_bps]
            hold_logit = symbol_action[self.symbol_max_orders + 3] * 5
            volume_signal = symbol_action[self.symbol_max_orders + 4]

            close_orders_probability = expit(close_orders_logit)
            hold_probability = expit(hold_logit)
            order_kind_prob = expit(order_kind_logit)
            tif_prob = expit(tif_logit)

            hold = bool(hold_probability > self.hold_threshold)
            modified_volume = self._get_modified_volume(symbol, volume_signal)

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
                'volume': volume_signal,
                'capacity': orders_capacity,
                'order_type': None,
                'modified_volume': modified_volume,
                'fee': float('nan'),
                'margin': float('nan'),
                'error': '',
            }
            if not hold and orders_capacity > 0:
                try:
                    est_margin = self._estimate_margin(symbol, modified_volume)
                    projected_margin = self.simulator.margin + est_margin
                    projected_leverage = projected_margin / (self.simulator.balance + 1e-6)
                except Exception:
                    projected_leverage = np.inf
                if projected_leverage > self.max_leverage:
                    orders_info[symbol]['error'] = 'max_leverage_exceeded'
                    continue
                order_type = OrderType.Buy if volume_signal > 0 else OrderType.Sell
                fee = self.fee if isinstance(self.fee, float) else self.fee(symbol)
                market_order = True if order_kind_prob >= 0.5 else False
                tif = 'IOC' if tif_prob >= 0.5 else 'FOK'
                max_offset_bps = 5.0
                limit_offset_bps = float(max_offset_bps * max(0.0, abs(limit_offset_norm)))
                try:
                    order = self.simulator.create_order(
                        order_type, symbol, modified_volume, fee,
                        market_order=market_order,
                        limit_offset_bps=limit_offset_bps,
                        tif=tif
                    )
                    orders_info[symbol].update({
                        'order_id': order.id,
                        'order_type': order_type,
                        'fee': fee,
                        'margin': order.margin,
                    })
                except ValueError as e:
                    orders_info[symbol]['error'] = str(e)

        return orders_info, closed_orders_info

    def _get_modified_volume(self, symbol: str, volume_signal: float) -> float:
        """Project a continuous volume signal [-1,1] to a feasible order volume.
        - Applies a small dead-zone to avoid tiny/accidental orders
        - Respects instrument volume_min/step/max
        - Scales down to honor max_leverage based on current margin
        Returns a non-negative volume (magnitude only)."""
        try:
            si = self.simulator.symbols_info[symbol]
            vol_min = float(si.volume_min)
            vol_max = float(si.volume_max)
            step = float(si.volume_step) if float(si.volume_step) > 0 else vol_min
        except Exception:
            # Fallback sensible defaults
            vol_min, vol_max, step = 0.01, 100.0, 0.01

        abs_sig = float(abs(volume_signal))
        # Dead-zone: treat small signals as no-trade
        if abs_sig < 0.05:
            return 0.0

        # Map signal to [vol_min, vol_max]
        raw_vol = vol_min + abs_sig * (vol_max - vol_min)

        # Enforce leverage cap by scaling down if needed
        try:
            est_margin = self._estimate_margin(symbol, raw_vol)
            # Max extra margin allowed
            margin_cap = max(0.0, self.max_leverage * (self.simulator.balance + 1e-6) - self.simulator.margin)
            if est_margin > margin_cap and est_margin > 0:
                scale = margin_cap / est_margin
                raw_vol *= max(0.0, min(1.0, scale))
        except Exception:
            pass

        # Quantize to volume_step (floor) and clip
        if step <= 0:
            step = vol_min
        quant_vol = float(np.floor(max(raw_vol, 0.0) / step) * step)
        quant_vol = float(min(max(quant_vol, 0.0), vol_max))

        # If below instrument minimum after constraints, skip order
        if quant_vol < vol_min:
            return 0.0
        return quant_vol

    def _estimate_margin(self, symbol: str, volume: float) -> float:
        si = self.simulator.symbols_info[symbol]
        price = float(self.simulator.price_at(symbol, self.simulator.current_time)['Close'])
        v = abs(volume) * si.trade_contract_size
        local_margin = (v * price) / self.simulator.leverage
        local_margin *= si.margin_rate
        unit_ratio = self.simulator._get_unit_ratio(symbol, self.simulator.current_time)
        return local_margin * unit_ratio

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
