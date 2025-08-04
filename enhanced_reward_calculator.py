"""
Enhanced Reward Engineering for Trading RL Agent

Key Improvements Recommended:
1. Better reward normalization
2. More sophisticated risk metrics
3. Market regime awareness
4. Position sizing optimization
"""

import numpy as np
from typing import Dict, List, Tuple

class EnhancedRewardCalculator:
    def __init__(self, config: Dict):
        self.config = config
        self.trade_history = []
        self.equity_history = []
        self.volatility_window = 20
        
    def calculate_enhanced_reward(self, env_state: Dict) -> float:
        """Calculate sophisticated reward combining multiple trading metrics"""
        
        equity_change = self._calculate_normalized_equity_change(env_state)
        
        risk_adjusted_return = self._calculate_risk_adjusted_return(env_state)
        
        drawdown_penalty = self._calculate_progressive_drawdown_penalty(env_state)
        
        trade_quality = self._calculate_trade_quality_score(env_state)
        
        regime_factor = self._calculate_market_regime_factor(env_state)
        
        sizing_bonus = self._calculate_position_sizing_bonus(env_state)
        
        weights = self._get_dynamic_weights(env_state)
        
        total_reward = (
            weights['equity'] * equity_change +
            weights['risk_adj'] * risk_adjusted_return +
            weights['drawdown'] * drawdown_penalty +
            weights['trade_quality'] * trade_quality +
            weights['sizing'] * sizing_bonus
        ) * regime_factor
        
        return np.clip(total_reward, -10.0, 10.0)
    
    def _calculate_normalized_equity_change(self, env_state: Dict) -> float:
        """Normalize equity change by recent volatility"""
        current_equity = env_state['equity']
        prev_equity = env_state.get('prev_equity', current_equity)
        
        if len(self.equity_history) < 2:
            return 0.0
            
        pct_change = (current_equity - prev_equity) / (prev_equity + 1e-6)
        
        recent_returns = np.diff(self.equity_history[-self.volatility_window:])
        volatility = np.std(recent_returns) + 1e-6
        
        return pct_change / volatility
    
    def _calculate_risk_adjusted_return(self, env_state: Dict) -> float:
        """Calculate Sharpe-like ratio for recent trades"""
        if len(self.trade_history) < 5:
            return 0.0
            
        recent_returns = [t['return'] for t in self.trade_history[-10:]]
        if len(recent_returns) < 2:
            return 0.0
            
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-6
        
        return mean_return / std_return
    
    def _calculate_progressive_drawdown_penalty(self, env_state: Dict) -> float:
        """Progressive penalty that increases exponentially with drawdown"""
        drawdown = env_state.get('drawdown', 0.0)
        
        if drawdown < 0.05:  # < 5% drawdown
            return 0.0
        elif drawdown < 0.10:  # 5-10% drawdown
            return -0.5 * (drawdown - 0.05) ** 2
        else:  # > 10% drawdown - severe penalty
            return -2.0 * (drawdown - 0.05) ** 2
    
    def _calculate_trade_quality_score(self, env_state: Dict) -> float:
        """Combined score of win rate and profit factor"""
        if len(self.trade_history) < 3:
            return 0.0
            
        recent_trades = self.trade_history[-20:]  # Last 20 trades
        profits = [t['profit'] for t in recent_trades]
        
        wins = sum(1 for p in profits if p > 0)
        win_rate = wins / len(profits)
        
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = gross_profit / (gross_loss + 1e-6)
        
        return 0.5 * (win_rate - 0.5) + 0.3 * np.log(profit_factor + 1e-6)
    
    def _calculate_market_regime_factor(self, env_state: Dict) -> float:
        """Adjust rewards based on market volatility regime"""
        market_volatility = env_state.get('market_volatility', 0.01)
        
        if market_volatility < 0.005:  # Low volatility
            return 0.8  # Reduce rewards (harder to profit)
        elif market_volatility < 0.02:  # Normal volatility
            return 1.0  # Normal rewards
        else:  # High volatility
            return 1.2  # Increase rewards (more opportunities)
    
    def _calculate_position_sizing_bonus(self, env_state: Dict) -> float:
        """Reward optimal position sizing (Kelly criterion inspired)"""
        position_size = env_state.get('position_size', 0.0)
        account_balance = env_state.get('balance', 1.0)
        
        if len(self.trade_history) < 5:
            optimal_size = 0.02  # Default 2%
        else:
            recent_trades = self.trade_history[-10:]
            win_rate = sum(1 for t in recent_trades if t['profit'] > 0) / len(recent_trades)
            avg_win = np.mean([t['return'] for t in recent_trades if t['profit'] > 0])
            avg_loss = abs(np.mean([t['return'] for t in recent_trades if t['profit'] < 0]))
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1-win_rate)) / avg_win
                optimal_size = np.clip(kelly_fraction, 0.01, 0.05)  # 1-5% max
            else:
                optimal_size = 0.02
        
        size_ratio = position_size / (account_balance + 1e-6)
        deviation = abs(size_ratio - optimal_size) / optimal_size
        
        return 0.1 * np.exp(-deviation)  # Exponential bonus for optimal sizing
    
    def _get_dynamic_weights(self, env_state: Dict) -> Dict[str, float]:
        """Dynamic reward component weights based on trading context"""
        market_volatility = env_state.get('market_volatility', 0.01)
        drawdown = env_state.get('drawdown', 0.0)
        
        weights = {
            'equity': 0.4,
            'risk_adj': 0.25,
            'drawdown': 0.15,
            'trade_quality': 0.15,
            'sizing': 0.05
        }
        
        if drawdown > 0.05:
            weights['drawdown'] *= 2.0
            weights['risk_adj'] *= 1.5
            weights['equity'] *= 0.7
        
        if market_volatility > 0.02:  # High volatility
            weights['trade_quality'] *= 1.5  # Emphasize quality
            weights['sizing'] *= 2.0  # Emphasize sizing
        
        total_weight = sum(weights.values())
        return {k: v/total_weight for k, v in weights.items()}
