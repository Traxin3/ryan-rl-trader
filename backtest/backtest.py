import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import PPO
from gym_mtsim.envs.mt_env import MtEnv
from gym_mtsim.simulator.order import OrderType
import matplotlib.pyplot as plt
import yaml
import os
import copy
from tqdm import tqdm

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
        
        print(f"🔧 Environment setup:")
        print(f"   Start tick: {test_env._start_tick}")
        print(f"   End tick: {test_env._end_tick}")
        print(f"   Current tick: {test_env._current_tick}")
        print(f"   Total time points: {len(test_env.time_points)}")
        
        print("🔄 Resetting environment...")
        obs, info = test_env.reset()
        done = False
        truncated = False
        
        total_steps = test_env._end_tick - test_env._start_tick
        current_step = 0
        
        print(f"🔄 Running backtest over {total_steps} time steps...")
        
        initial_equity = info.get('equity', test_env.simulator.balance)
        max_equity = initial_equity
        min_equity = initial_equity
        
        with tqdm(total=total_steps, desc="Backtest Progress", unit="steps", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            try:
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = test_env.step(action)
                    
                    current_step += 1
                    pbar.update(1)
                    
                    current_equity = info.get('equity', 0)
                    max_equity = max(max_equity, current_equity)
                    min_equity = min(min_equity, current_equity)
                    
                    equity_change = ((current_equity - initial_equity) / initial_equity) * 100
                    max_drawdown = ((max_equity - current_equity) / max_equity) * 100 if max_equity > 0 else 0
                    
                    if current_step % 50 == 0:  # Update every 50 steps for smoother display
                        pbar.set_postfix({
                            'Equity': f"{current_equity:.2f}",
                            'P&L%': f"{equity_change:+.2f}%",
                            'DD%': f"{max_drawdown:.2f}%",
                            'Trades': info.get('total_trades', 0),
                            'Win%': f"{info.get('win_rate', 0)*100:.1f}%",
                            'Balance': f"{info.get('balance', 0):.0f}"
                        })
                    
                    if current_step >= total_steps * 2:  # Allow 2x buffer
                        print("⚠️ Backtest exceeded expected steps, stopping...")
                        break
                        
            except KeyboardInterrupt:
                print("\n⚠️ Backtest interrupted by user")
                raise
            except Exception as e:
                print(f"\n❌ Error during backtest: {e}")
                raise
            
        final_equity = info.get('equity', 0)
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
        final_drawdown = ((max_equity - final_equity) / max_equity) * 100 if max_equity > 0 else 0
        
        print(f"\n🎯 Backtest Summary:")
        print(f"   📈 Final Equity: {final_equity:.2f} (Start: {initial_equity:.2f})")
        print(f"   💰 Total Return: {total_return:+.2f}%")
        print(f"   📊 Max Equity: {max_equity:.2f}")
        print(f"   📉 Max Drawdown: {final_drawdown:.2f}%")
        print(f"   🎲 Total Trades: {info.get('total_trades', 0)}")
        print(f"   🏆 Win Rate: {info.get('win_rate', 0)*100:.1f}%")
        
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
            
            if (current_symbols != test_env.symbols or 
                current_timeframes != test_env.timeframes):
                print(f"📊 Updating environment for symbol: {current_symbols}, timeframe: {current_timeframes}")
                test_env.symbols = current_symbols
                test_env.timeframes = current_timeframes
                test_env.use_cached_features = True
                test_env._process_data()
            else:
                print("✅ Using existing environment configuration")
            
        return test_env
    
    def _analyze_with_vectorbt(self, env: MtEnv) -> Dict:
        """
        Convert MT5 simulator results to vectorbt format and analyze
        
        Args:
            env: MtEnv instance with completed backtest
            
        Returns:
            Dictionary of comprehensive metrics and results
        """
        print("📊 Analyzing backtest results...")
        
        symbol = env.symbols[0]
        timeframe = env.timeframes[0]
        df = env.simulator.symbols_data[(symbol, timeframe)]
        price = df['Close']
        
        print("🔍 Extracting trades...")
        trades = self._extract_trades(env)
        
        print("📈 Creating portfolio analysis...")
        pf = self._create_vectorbt_portfolio(price, trades)
        
        print("📋 Calculating performance metrics...")
        metrics = self._calculate_metrics(pf, trades)
        
        print("📊 Generating visualizations...")
        self._generate_visualizations(pf, price, trades)
        
        self.results = metrics
        return metrics
    
    def _extract_trades(self, env: MtEnv) -> List[Dict]:
        """Extract trades from environment history"""
        trades = []
        
        if not hasattr(env, 'trade_history') or not env.trade_history:
            print("⚠️ No trade history found in environment")
            return trades
            
        for trade in env.trade_history:
            try:
                if isinstance(trade, dict):
                    order_type = trade.get('order_type', trade.get('type'))
                    
                    if hasattr(order_type, 'name'):
                        direction = 'Long' if order_type.name.lower() in ['buy', 'long'] else 'Short'
                    elif isinstance(order_type, str):
                        direction = 'Long' if order_type.lower() in ['buy', 'long'] else 'Short'
                    elif isinstance(order_type, int):
                        direction = 'Long' if order_type == 0 else 'Short'  # Assuming 0=Buy, 1=Sell
                    else:
                        direction = 'Long'  # Default fallback
                    
                    trades.append({
                        'EntryTime': trade.get('open_time', trade.get('entry_time')),
                        'ExitTime': trade.get('close_time', trade.get('exit_time')),
                        'EntryPrice': trade.get('entry_price', trade.get('open_price')),
                        'ExitPrice': trade.get('exit_price', trade.get('close_price')),
                        'Size': trade.get('volume', trade.get('size', 1.0)),
                        'PnL': trade.get('profit', trade.get('pnl', 0.0)),
                        'Return': trade.get('profit', 0.0) / (trade.get('margin', 1.0) + 1e-6),
                        'Direction': direction,
                        'Symbol': trade.get('symbol', env.symbols[0] if env.symbols else 'UNKNOWN')
                    })
            except Exception as e:
                print(f"⚠️ Error processing trade: {e}")
                continue
                
        print(f"📊 Extracted {len(trades)} trades from environment")
        return trades
    
    def _create_vectorbt_portfolio(self, price: pd.Series, trades: List[Dict]) -> vbt.Portfolio:
        """Create vectorbt Portfolio from trades"""
        if not trades:
            print("⚠️ No trades found, creating buy-and-hold portfolio for comparison")
            return vbt.Portfolio.from_holding(
                price,
                init_cash=self.env.simulator.balance,
                fees=self.env.fee if hasattr(self.env, 'fee') and isinstance(self.env.fee, float) else 0.0005,
                freq='1min'
            )
        
        trades_df = pd.DataFrame(trades)
        print(f"🔍 Processing {len(trades_df)} trades for VectorBT...")
        
        trades_df['EntryTime'] = pd.to_datetime(trades_df['EntryTime'])
        trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'])
        
        print(f"📊 Price data range: {price.index[0]} to {price.index[-1]}")
        print(f"📊 Trade time range: {trades_df['EntryTime'].min()} to {trades_df['ExitTime'].max()}")
        
        entries = pd.Series(False, index=price.index)
        exits = pd.Series(False, index=price.index)
        short_entries = pd.Series(False, index=price.index)
        short_exits = pd.Series(False, index=price.index)
        
        mapped_entries = 0
        mapped_exits = 0
        
        for _, trade in trades_df.iterrows():
            try:
                entry_time = trade['EntryTime']
                exit_time = trade['ExitTime']
                
                try:
                    entry_idx = price.index.get_indexer([entry_time], method='nearest')[0]
                    exit_idx = price.index.get_indexer([exit_time], method='nearest')[0]
                    
                    if entry_idx >= 0 and exit_idx >= 0:
                        entry_ts = price.index[entry_idx]
                        exit_ts = price.index[exit_idx]
                        
                        if trade['Direction'] == 'Long':
                            entries.loc[entry_ts] = True
                            exits.loc[exit_ts] = True
                            mapped_entries += 1
                        else:  # Short
                            short_entries.loc[entry_ts] = True
                            short_exits.loc[exit_ts] = True
                            mapped_entries += 1
                        mapped_exits += 1
                except (KeyError, IndexError):
                    if entry_time in price.index and exit_time in price.index:
                        if trade['Direction'] == 'Long':
                            entries.loc[entry_time] = True
                            exits.loc[exit_time] = True
                        else:  # Short
                            short_entries.loc[entry_time] = True
                            short_exits.loc[exit_time] = True
                        mapped_entries += 1
                        mapped_exits += 1
                        
            except Exception as e:
                print(f"⚠️ Error mapping trade: {e}")
                continue
        
        print(f"✅ Mapped {mapped_entries} entry signals and {mapped_exits} exit signals")
        print(f"📊 Long entries: {entries.sum()}, Long exits: {exits.sum()}")
        print(f"📊 Short entries: {short_entries.sum()}, Short exits: {short_exits.sum()}")
        
        pf = vbt.Portfolio.from_signals(
            price,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            direction='both',
            fees=self.env.fee if hasattr(self.env, 'fee') and isinstance(self.env.fee, float) else 0.0005,
            freq='1T',  # 1 minute frequency
            init_cash=self.env.simulator.balance,
            slippage=0.001,  # 0.1% slippage
            size=1.0,  # Fixed size for now
            accumulate=False,  # Don't accumulate positions
        )
        
        print(f"💼 Portfolio created with {pf.orders.count()} orders")
        return pf
    
    def _calculate_metrics(self, pf: vbt.Portfolio, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        def safe_metric(func, fallback=0.0):
            try:
                result = func()
                return result if not pd.isna(result) else fallback
            except:
                return fallback
        
        metrics = {
            'total_return': safe_metric(lambda: pf.total_return()),
            'annualized_return': safe_metric(lambda: pf.annualized_return()),
            'sharpe_ratio': safe_metric(lambda: pf.sharpe_ratio()),
            'sortino_ratio': safe_metric(lambda: pf.sortino_ratio()),
            'max_drawdown': safe_metric(lambda: pf.max_drawdown()),
            'win_rate': safe_metric(lambda: pf.win_rate()),
            'profit_factor': safe_metric(lambda: pf.profit_factor(), 1.0),
            'expectancy': safe_metric(lambda: pf.expectancy()),
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['PnL'] > 0]) if not trades_df.empty else 0,
            'losing_trades': len(trades_df[trades_df['PnL'] < 0]) if not trades_df.empty else 0,
            'avg_trade_duration': (trades_df['ExitTime'] - trades_df['EntryTime']).mean() if not trades_df.empty else pd.Timedelta(0),
            'equity_curve': pf.value(),
            'underwater': pf.drawdown(),
            'trades': trades_df
        }
        
        return metrics
    
    def _generate_visualizations(self, pf: vbt.Portfolio, price: pd.Series, trades: List[Dict]):
        """Generate interactive visualizations of backtest results"""
        try:
            print(f"📊 Creating visualizations for portfolio with {pf.orders.count()} orders...")
            
            fig = pf.plot(subplots=[
                'orders',
                'positions', 
                'trade_pnl',
                'cum_returns',
                'drawdowns'
            ])
            fig.update_layout(
                title=f"Backtest Results - {len(trades)} Trades", 
                height=1200,
                showlegend=True
            )
            
            fig.show()
            
            if not os.path.exists('backtest_results'):
                os.makedirs('backtest_results')
            fig.write_html("backtest_results/performance.html")
            print("✅ Performance chart saved to backtest_results/performance.html")
            
            if trades and pf.orders.count() > 0:  
                trades_df = pd.DataFrame(trades)
                
                try:
                    trade_fig = pf.trades.plot()
                    trade_fig.update_layout(
                        title=f"Individual Trade Analysis - {len(trades)} Trades",
                        height=800
                    )
                    trade_fig.write_html("backtest_results/trade_analysis.html")
                    print(f"✅ Trade analysis saved with {len(trades)} trades")
                except Exception as e:
                    print(f"⚠️ Could not create trade analysis plot: {e}")
                
                trades_df.to_csv('backtest_results/trades.csv', index=False)
                print(f"✅ Trade data saved to backtest_results/trades.csv")
                
                if pf.orders.count() == 0:
                    print("⚠️ No VBT orders found, creating manual trade overlay...")
                    self._create_manual_trade_plot(price, trades_df)
                    
            else:
                print("⚠️ No trades to analyze - portfolio may not have executed properly")
                
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            pass
    
    def _create_manual_trade_plot(self, price: pd.Series, trades_df: pd.DataFrame):
        """Create manual trade overlay plot when VBT doesn't show trades properly"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price with Trade Markers', 'Cumulative PnL'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=price.index,
                    y=price.values,
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            for _, trade in trades_df.iterrows():
                try:
                    entry_time = pd.to_datetime(trade['EntryTime'])
                    exit_time = pd.to_datetime(trade['ExitTime'])
                    
                    if entry_time in price.index:
                        entry_price = price.loc[entry_time]
                    else:
                        entry_idx = price.index.get_indexer([entry_time], method='nearest')[0]
                        entry_price = price.iloc[entry_idx] if entry_idx >= 0 else trade['EntryPrice']
                    
                    if exit_time in price.index:
                        exit_price = price.loc[exit_time]
                    else:
                        exit_idx = price.index.get_indexer([exit_time], method='nearest')[0]
                        exit_price = price.iloc[exit_idx] if exit_idx >= 0 else trade['ExitPrice']
                    
                    color = 'green' if trade['Direction'] == 'Long' else 'red'
                    fig.add_trace(
                        go.Scatter(
                            x=[entry_time],
                            y=[entry_price],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up' if trade['Direction'] == 'Long' else 'triangle-down',
                                size=8,
                                color=color
                            ),
                            name=f"{trade['Direction']} Entry",
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[exit_time],
                            y=[exit_price],
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=8,
                                color='black'
                            ),
                            name='Exit',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                except Exception as e:
                    continue
            
            trades_df_sorted = trades_df.sort_values('EntryTime')
            cum_pnl = trades_df_sorted['PnL'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=trades_df_sorted['ExitTime'],
                    y=cum_pnl,
                    mode='lines+markers',
                    name='Cumulative PnL',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"Manual Trade Analysis - {len(trades_df)} Trades",
                height=800,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative PnL", row=2, col=1)
            
            fig.write_html("backtest_results/manual_trade_analysis.html")
            print("✅ Manual trade analysis saved to backtest_results/manual_trade_analysis.html")
            
        except Exception as e:
            print(f"⚠️ Error creating manual trade plot: {e}")
    
    def generate_report(self, output_path: str = 'backtest_results/report.html'):
        """Generate simple text summary of backtest results"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        summary_text = f"""
Backtest Results Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Total Return: {self.results['total_return']:.2%}
- Annualized Return: {self.results['annualized_return']:.2%}
- Sharpe Ratio: {self.results['sharpe_ratio']:.3f}
- Sortino Ratio: {self.results['sortino_ratio']:.3f}
- Max Drawdown: {self.results['max_drawdown']:.2%}
- Win Rate: {self.results['win_rate']:.2%}
- Profit Factor: {self.results['profit_factor']:.2f}

Trade Statistics:
- Total Trades: {self.results['total_trades']}
- Winning Trades: {self.results['winning_trades']}
- Losing Trades: {self.results['losing_trades']}
"""
        
        summary_path = output_path.replace('.html', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"📄 Summary saved to {summary_path}")
        return summary_path
