import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
from ..simulator.order import OrderType

class MarketImpactModel:
    """
    Realistic market impact and liquidity modeling for trading simulation.
    Implements slippage, bid-ask spreads, and liquidity-dependent execution.
    """
    
    def __init__(self, 
                 base_spread_bps: float = 0.5,  # Base spread in basis points
                 slippage_factor: float = 0.1,   # Slippage multiplier
                 liquidity_impact: float = 0.05, # Impact per volume unit
                 spread_volatility_factor: float = 2.0):
        self.base_spread_bps = base_spread_bps
        self.slippage_factor = slippage_factor
        self.liquidity_impact = liquidity_impact
        self.spread_volatility_factor = spread_volatility_factor
        
        self.session_spreads = {
            'asian': 1.2,      # Higher spreads during Asian session
            'london': 0.7,     # Tighter spreads during London
            'ny': 0.8,         # Moderate spreads during NY
            'overlap': 0.5,    # Tightest during overlaps
            'off_hours': 2.0   # Widest during off hours
        }
        
        self.volatility_buckets = {
            'low': 0.8,        # < 0.5% hourly volatility
            'normal': 1.0,     # 0.5-1.5% hourly volatility  
            'high': 1.5,       # 1.5-3% hourly volatility
            'extreme': 3.0     # > 3% hourly volatility
        }

    def get_trading_session(self, dt: datetime) -> str:
        """Determine current trading session"""
        hour = dt.hour
        
        if 8 <= hour < 13:
            return 'london'
        elif 13 <= hour < 17:
            return 'overlap'  # London-NY overlap
        elif 17 <= hour < 22:
            return 'ny'
        elif 22 <= hour <= 23 or 0 <= hour < 8:
            return 'asian'
        else:
            return 'off_hours'

    def calculate_volatility_regime(self, price_data: np.ndarray, window: int = 24) -> str:
        """Calculate current volatility regime"""
        if len(price_data) < 2:
            return 'normal'
            
        returns = np.diff(np.log(price_data[-window:]))
        if len(returns) == 0:
            return 'normal'
            
        hourly_vol = np.std(returns) * np.sqrt(24)  # Annualized hourly vol
        
        if hourly_vol < 0.005:
            return 'low'
        elif hourly_vol < 0.015:
            return 'normal'
        elif hourly_vol < 0.03:
            return 'high'
        else:
            return 'extreme'

    def get_dynamic_spread(self, 
                          symbol: str,
                          current_time: datetime,
                          price_data: np.ndarray,
                          current_price: float) -> Tuple[float, float]:
        """
        Calculate dynamic bid-ask spread based on market conditions
        Returns (bid_price, ask_price)
        """
        base_spread = self.base_spread_bps / 10000 * current_price
        
        session = self.get_trading_session(current_time)
        session_multiplier = self.session_spreads.get(session, 1.0)
        
        vol_regime = self.calculate_volatility_regime(price_data)
        vol_multiplier = self.volatility_buckets.get(vol_regime, 1.0)
        
        hour = current_time.hour
        minute = current_time.minute
        
        time_adjustment = 1.0
        session_transition_hours = [8, 13, 17, 22]  # Major session times
        for session_hour in session_transition_hours:
            time_diff = abs((hour * 60 + minute) - (session_hour * 60))
            if time_diff <= 30:  # 30 minutes window
                time_adjustment = max(time_adjustment, 1.5)
        
        total_spread = base_spread * session_multiplier * vol_multiplier * time_adjustment
        half_spread = total_spread / 2
        
        bid = current_price - half_spread
        ask = current_price + half_spread
        
        return bid, ask

    def calculate_slippage(self,
                          order_type: OrderType,
                          volume: float,
                          current_price: float,
                          volatility: float,
                          liquidity_score: float = 1.0) -> float:
        """
        Calculate realistic slippage based on volume, volatility, and liquidity
        Returns the slippage amount (positive = adverse movement)
        """
        volume_impact = self.liquidity_impact * np.log1p(volume) / liquidity_score
        
        volatility_impact = volatility * self.slippage_factor
        
        random_component = np.random.normal(0, volatility_impact * 0.5)
        
        total_slippage = (volume_impact + volatility_impact + abs(random_component)) * current_price
        
        if order_type == OrderType.Buy:
            return total_slippage  # Buy at higher price
        else:
            return -total_slippage  # Sell at lower price

    def get_execution_price(self,
                           order_type: OrderType,
                           volume: float,
                           symbol: str,
                           current_time: datetime,
                           price_data: np.ndarray,
                           current_price: float,
                           market_order: bool = True) -> Tuple[float, Dict]:
        """
        Calculate realistic execution price including spreads and slippage
        
        Returns:
            execution_price: Final execution price
            execution_info: Dict with breakdown of costs
        """
        bid, ask = self.get_dynamic_spread(symbol, current_time, price_data, current_price)
        
        volatility = self.calculate_volatility_regime(price_data)
        vol_value = np.std(np.diff(np.log(price_data[-24:]))) if len(price_data) > 24 else 0.01
        
        session = self.get_trading_session(current_time)
        liquidity_score = {
            'london': 1.2,
            'ny': 1.1, 
            'overlap': 1.5,
            'asian': 0.8,
            'off_hours': 0.5
        }.get(session, 1.0)
        
        if market_order:
            if order_type == OrderType.Buy:
                base_price = ask  # Buy at ask
            else:
                base_price = bid  # Sell at bid
                
            slippage = self.calculate_slippage(
                order_type, volume, current_price, vol_value, liquidity_score
            )
            execution_price = base_price + slippage
            
        else:
            execution_price = current_price  # Simplified for now
            slippage = 0
        
        execution_info = {
            'base_price': current_price,
            'bid': bid,
            'ask': ask,
            'spread_bps': ((ask - bid) / current_price) * 10000,
            'slippage': slippage,
            'slippage_bps': (slippage / current_price) * 10000,
            'execution_price': execution_price,
            'session': session,
            'volatility_regime': volatility,
            'liquidity_score': liquidity_score,
            'total_cost_bps': ((execution_price - current_price) / current_price) * 10000 * order_type.sign
        }
        
        return execution_price, execution_info

    def get_daily_liquidity_pattern(self, hour: int) -> float:
        """Get relative liquidity level by hour (24h pattern)"""
        liquidity_pattern = {
            0: 0.3, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5,  # Asian late/quiet
            6: 0.6, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.0,  # London open
            12: 1.0, 13: 1.2, 14: 1.3, 15: 1.4, 16: 1.3,      # London-NY overlap
            17: 1.2, 18: 1.0, 19: 0.9, 20: 0.8, 21: 0.7,      # NY session
            22: 0.6, 23: 0.4                                    # Quiet period
        }
        return liquidity_pattern.get(hour, 0.5)
