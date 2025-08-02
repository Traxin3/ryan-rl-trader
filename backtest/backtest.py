import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import PPO
from gym_mtsim.envs.mt_env import MtEnv
import matplotlib.pyplot as plt
import yaml
import os

class VectorBTBacktester:
    def __init__(self, env: MtEnv, model_path: str = 'ppo_transformer_mtsim_final'):
        """
        Initialize the backtester with environment and trained model
        
        Args:
            env: MtEnv - Trading environment instance
            model_path: str - Path to trained PPO model
        """
        self.env = env
        self.model = PPO.load(model_path)
        self.results = {}
        
    def run_backtest(
        self,
        test_period: Optional[Tuple[datetime, datetime]] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[int] = None
    ) -> Dict:
        """
        Run backtest on specified period and symbol/timeframe
        
        Args:
            test_period: Optional time range for backtest
            symbol: Optional specific symbol to backtest
            timeframe: Optional specific timeframe to backtest
            
        Returns:
            Dictionary of backtest results and metrics
        """
        test_env = self._prepare_test_env(test_period, symbol, timeframe)
        
        obs = test_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, _ = test_env.step(action)
            
        return self._analyze_with_vectorbt(test_env)
    
    def _prepare_test_env(
        self,
        test_period: Optional[Tuple[datetime, datetime]] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[int] = None
    ) -> MtEnv:
        """
        Prepare test environment with specified parameters
        
        Args:
            test_period: Optional time range
            symbol: Optional specific symbol
            timeframe: Optional specific timeframe
            
        Returns:
            Configured MtEnv instance
        """
        test_env = copy.deepcopy(self.env)
        
        if test_period:
            test_env.simulator.set_time_range(test_period[0], test_period[1])
            
        if symbol or timeframe:
            current_symbols = [symbol] if symbol else test_env.symbols
            current_timeframes = [timeframe] if timeframe else test_env.timeframes
            test_env.symbols = current_symbols
            test_env.timeframes = current_timeframes
            test_env._process_data()  # Reprocess data with new symbols/timeframes
            
        return test_env
    
    def _analyze_with_vectorbt(self, env: MtEnv) -> Dict:
        """
        Convert MT4 simulator results to vectorbt format and analyze
        
        Args:
            env: MtEnv instance with completed backtest
            
        Returns:
            Dictionary of comprehensive metrics and results
        """
        symbol = env.symbols[0]
        timeframe = env.timeframes[0]
        df = env.simulator.symbols_data[(symbol, timeframe)]
        price = df['Close']
        
        trades = self._extract_trades(env)
        
        pf = self._create_vectorbt_portfolio(price, trades)
        
        metrics = self._calculate_metrics(pf, trades)
        
        self._generate_visualizations(pf, price, trades)
        
        self.results = metrics
        return metrics
    
    def _extract_trades(self, env: MtEnv) -> List[Dict]:
        """Extract trades from environment history"""
        trades = []
        for trade in env.trade_history:
            trades.append({
                'EntryTime': trade['open_time'],
                'ExitTime': trade['close_time'],
                'EntryPrice': trade['entry_price'],
                'ExitPrice': trade['exit_price'],
                'Size': trade['volume'],
                'PnL': trade['profit'],
                'Return': trade['profit'] / (trade['margin'] + 1e-6),
                'Direction': 'Long' if trade['order_type'] == OrderType.Buy else 'Short',
                'Symbol': trade['symbol']
            })
        return trades
    
    def _create_vectorbt_portfolio(self, price: pd.Series, trades: List[Dict]) -> vbt.Portfolio:
        """Create vectorbt Portfolio from trades"""
        trades_df = pd.DataFrame(trades)
        
        long_trades = trades_df[trades_df['Direction'] == 'Long']
        short_trades = trades_df[trades_df['Direction'] == 'Short']
        
        pf = vbt.Portfolio.from_signals(
            price,
            entries=long_trades['EntryTime'],
            exits=long_trades['ExitTime'],
            short_entries=short_trades['EntryTime'],
            short_exits=short_trades['ExitTime'],
            direction='both',
            fees=self.env.fee if isinstance(self.env.fee, float) else 0.0005,
            freq='1m',
            init_cash=self.env.simulator.balance,
            slippage=0.001,  # 0.1% slippage
        )
        
        return pf
    
    def _calculate_metrics(self, pf: vbt.Portfolio, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        trades_df = pd.DataFrame(trades)
        
        metrics = {
            'total_return': pf.total_return(),
            'annualized_return': pf.annualized_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'sortino_ratio': pf.sortino_ratio(),
            'max_drawdown': pf.max_drawdown(),
            'win_rate': pf.win_rate(),
            'profit_factor': pf.profit_factor(),
            'expectancy': pf.expectancy(),
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['PnL'] > 0]),
            'losing_trades': len(trades_df[trades_df['PnL'] < 0]),
            'avg_trade_duration': (trades_df['ExitTime'] - trades_df['EntryTime']).mean(),
            'equity_curve': pf.value(),
            'underwater': pf.drawdown(),
            'trades': trades_df
        }
        
        return metrics
    
    def _generate_visualizations(self, pf: vbt.Portfolio, price: pd.Series, trades: pd.DataFrame):
        """Generate interactive visualizations of backtest results"""
        fig = pf.plot(subplots=[
            'orders',
            'trade_pnl',
            'cum_returns',
            'drawdowns'
        ])
        fig.update_layout(title="Backtest Results", height=1000)
        fig.show()
        
        if not os.path.exists('backtest_results'):
            os.makedirs('backtest_results')
        fig.write_html("backtest_results/performance.html")
        
        if not trades.empty:
            trade_fig = pf.trades.plot()
            trade_fig.update_layout(title="Trade Analysis")
            trade_fig.write_html("backtest_results/trade_analysis.html")
    
    def generate_report(self, output_path: str = 'backtest_results/report.html'):
        """Generate comprehensive HTML report"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
            
        report = vbt.HTMLReport(
            title="RL Trading Strategy Backtest Report",
            description=f"Backtest results for trained model on {datetime.now().date()}"
        )
        
        report.add_header("Performance Metrics")
        report.add_text("Key performance indicators:")
        report.add_dict({
            k: v for k, v in self.results.items() 
            if k not in ['equity_curve', 'underwater', 'trades']
        })
        
        report.add_header("Equity Curve")
        report.add_figure(self.results['equity_curve'].vbt.plot().figure)
        
        if not self.results['trades'].empty:
            report.add_header("Trade Analysis")
            report.add_dataframe(self.results['trades'].head(50))
            
        report.save(output_path)
        print(f"Report saved to {output_path}")
