import os
import platform
import yaml
import argparse
import subprocess
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from gym_mtsim.envs.mt_env import MtEnv
import warnings

warnings.filterwarnings('ignore')


def set_global_seeds(seed: int = 42):
    try:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        os.environ['PYTHONHASHSEED'] = str(seed)
    except Exception:
        pass

def colorize_performance(value, threshold_good=0.05, threshold_bad=-0.05):
    """Add color to performance metrics based on value"""
    if value > threshold_good:
        return f"🟢 {value:.2%}"
    elif value < threshold_bad:
        return f"🔴 {value:.2%}"
    else:
        return f"🟡 {value:.2%}"

def colorize_ratio(value, threshold_good=1.0, threshold_bad=0.5):
    """Add color to ratio metrics"""
    if value > threshold_good:
        return f"🟢 {value:.3f}"
    elif value < threshold_bad:
        return f"🔴 {value:.3f}"
    else:
        return f"🟡 {value:.3f}"

def find_npm():
    """Find npm executable with cross-platform support (Windows, Linux, MacOS)"""
    npm_paths = []
    
    system = platform.system().lower()
    if system == 'windows':
        npm_paths.extend([
            "npm",
            "npm.cmd",
            os.path.expanduser("~\\AppData\\Roaming\\npm\\npm.cmd"),
            os.path.expanduser("~\\AppData\\Local\\npm\\npm.cmd"),
            "C:\\Program Files\\nodejs\\npm.cmd",
            "C:\\Program Files (x86)\\nodejs\\npm.cmd"
        ])
    else:  # Linux and MacOS
        npm_paths.extend([
            "npm",
            "/usr/local/bin/npm",
            "/usr/bin/npm",
            os.path.expanduser("~/.nvm/versions/node/*/bin/npm"),
            os.path.expanduser("~/.npm-packages/bin/npm"),
            os.path.expanduser("~/node_modules/.bin/npm")
        ])
    
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for dir in path_dirs:
        if system == 'windows':
            npm_paths.append(os.path.join(dir, 'npm.cmd'))
        else:
            npm_paths.append(os.path.join(dir, 'npm'))
    
    tested_paths = set()
    for path in npm_paths:
        if '*' in path:
            import glob
            expanded_paths = glob.glob(path)
            for expanded in expanded_paths:
                if expanded not in tested_paths:
                    tested_paths.add(expanded)
                    try:
                        result = subprocess.run([expanded, "--version"], 
                                              capture_output=True, 
                                              text=True)
                        if result.returncode == 0:
                            print(f"✅ Found npm at: {expanded}")
                            return expanded
                    except:
                        continue
        else:
            if path not in tested_paths and os.path.exists(path):
                tested_paths.add(path)
                try:
                    result = subprocess.run([path, "--version"], 
                                          capture_output=True, 
                                          text=True)
                    if result.returncode == 0:
                        print(f"✅ Found npm at: {path}")
                        return path
                except:
                    continue
    
    print("❌ npm not found in any standard locations")
    print("💡 Please install Node.js from https://nodejs.org/")
    return None

def load_config(config_file: str = "config/config.yaml"):
    """Load configuration from YAML file with fallback paths"""
    config_paths = [
        config_file,
        os.path.join(os.path.dirname(__file__), config_file),
        "/content/ryan-rl-trader/config/config.yaml"  # Google Colab path
    ]
    
    for path in config_paths:
        try:
            with open(path) as f:
                print(f"✅ Successfully loaded config from: {path}")
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError(
        f"❌ Could not find config file in any of these locations:\n"
        f"{config_paths}\n"
        f"Please ensure the config file exists in one of these paths."
    )

def load_backtest_config():
    """Load backtest-specific configuration with fallback"""
    try:
        return load_config("config/backtest_config.yaml")
    except FileNotFoundError:
        print("⚠️ No backtest_config.yaml found, using defaults")
        return {
            'backtest': {
                'model_path': "ppo_transformer_mtsim_final",
                'output': {
                    'report_path': "backtest_results/report.html",
                    'save_trades': True,
                    'save_equity_curve': True
                }
            }
        }

def create_env(config, test_mode: bool = False, use_cached_features: bool = False):
    """Create trading environment from config with optional test mode settings"""
    env_config = config['env']
    
    reward_params = {
        'reward_scaling': env_config.get('reward_scaling', 2.0),
        'min_reward': env_config.get('min_reward', -10.0),
        'max_reward': env_config.get('max_reward', 10.0),
        'survival_bonus': env_config.get('survival_bonus', 0.02),
        'leverage_penalty': env_config.get('leverage_penalty', 0.02),
        'trade_reward_multiplier': env_config.get('trade_reward_multiplier', 2.0),
        'drawdown_penalty': env_config.get('drawdown_penalty', 0.2),
        'early_close_bonus': env_config.get('early_close_bonus', 0.1),
        'diversification_bonus': env_config.get('diversification_bonus', 0.05),
        'volatility_penalty': env_config.get('volatility_penalty', 0.05),
        'position_size_penalty': env_config.get('position_size_penalty', 0.02),
        'min_tp_sl_ratio': env_config.get('min_tp_sl_ratio', 2.0),
        'atr_multiplier': env_config.get('atr_multiplier', 1.5)
    }
    
    if test_mode:
        return MtEnv(
            symbols=env_config['symbols'],
            timeframes=env_config['timeframes'],
            window_size=env_config['window_size'],
            max_leverage=env_config['max_leverage'],
            reward_scaling=1.0, 
            risk_adjusted_reward=False,  
            use_cached_features=use_cached_features,
            **{k: v for k, v in reward_params.items() if k not in ['reward_scaling']}
        )
    
    return MtEnv(
        symbols=env_config['symbols'],
        timeframes=env_config['timeframes'],
        window_size=env_config['window_size'],
        max_leverage=env_config['max_leverage'],
        risk_adjusted_reward=env_config.get('risk_adjusted_reward', True),
        use_cached_features=use_cached_features,
        **reward_params
    )

def run_backtest():
    """Execute backtest using configured parameters"""
    print("\n" + "="*50)
    print("🚀 Starting Backtest")
    print("="*50)
    
    try:
        config = load_config()
        backtest_config = load_backtest_config()['backtest']
        
        print("📊 Creating backtest environment...")
        env = create_env(config, test_mode=True, use_cached_features=True)
        
        print(f"🤖 Loading model from: {backtest_config['model_path']}")
        from backtest.backtest import VectorBTBacktester
        backtester = VectorBTBacktester(env, backtest_config['model_path'])
        
        test_period = None
        if backtest_config.get('test_start') and backtest_config.get('test_end'):
            try:
                test_period = (
                    datetime.strptime(backtest_config['test_start'], '%Y-%m-%d'),
                    datetime.strptime(backtest_config['test_end'], '%Y-%m-%d')
                )
                print(f"📅 Backtest period: {test_period[0]} to {test_period[1]}")
            except ValueError as e:
                print(f"❌ Error parsing date range: {e}")
                print("⚠️ Using full dataset period instead")
        
        symbol = backtest_config.get('symbol')
        timeframe = backtest_config.get('timeframe')
        if symbol:
            print(f"📊 Backtesting specific symbol: {symbol}")
        if timeframe:
            print(f"⏱️ Backtesting specific timeframe: {timeframe}min")
        
        print("\n🏃 Running backtest...")
        start_time = time.time()
        results = backtester.run_backtest(
            test_period=test_period,
            symbol=symbol,
            timeframe=timeframe
        )
        
        report_path = backtest_config['output']['report_path']
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        actual_report_path = backtester.generate_report(report_path)
        
        duration = time.time() - start_time
        print("\n" + "="*60)
        print("✅ BACKTEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"⏱️  Execution Time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        print("="*60)
        
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        print(f"📈 Total Return:       {colorize_performance(results['total_return'])}")
        print(f"📅 Annualized Return:  {colorize_performance(results['annualized_return'])}")
        print(f"⚡ Sharpe Ratio:       {colorize_ratio(results['sharpe_ratio'])}")
        print(f"🛡️  Sortino Ratio:      {colorize_ratio(results['sortino_ratio'])}")
        print(f"📉 Max Drawdown:       {colorize_performance(-results['max_drawdown'], -0.05, -0.20)}")
        print(f"🎯 Win Rate:           {colorize_performance(results['win_rate'], 0.6, 0.4)}")
        print(f"💎 Profit Factor:      {colorize_ratio(results['profit_factor'], 1.5, 1.0)}")
        print(f"🔢 Total Trades:       {results['total_trades']}")
        print(f"🏆 Winning Trades:     {results['winning_trades']}")
        print(f"❌ Losing Trades:      {results['losing_trades']}")
        
        if results['total_trades'] > 0:
            avg_trade_return = results['total_return'] / results['total_trades'] if results['total_trades'] > 0 else 0
            print(f"📊 Avg Trade Return:   {colorize_performance(avg_trade_return, 0.01, -0.01)}")
        
        print("\n📂 RESULTS SAVED TO:")
        print("-" * 40)
        print(f"📄 Summary Report:     {os.path.abspath(actual_report_path)}")
        print(f"📊 Performance Chart:  {os.path.abspath('backtest_results/performance.html')}")
        if results['total_trades'] > 0:
            print(f"🔍 Trade Analysis:     {os.path.abspath('backtest_results/trade_analysis.html')}")
            print(f"📋 Trade Data:        {os.path.abspath('backtest_results/trades.csv')}")
        print("="*60)
        
    except Exception as e:
        print("\n❌ Backtest Failed:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def start_gui_mode():
    """Start the complete RL Trading Dashboard with backend and frontend"""
    print("\n" + "="*50)
    print("🚀 Starting RL Trading Dashboard")
    print("="*50)
    
    dash_path = Path("ryan-dash")
    if not dash_path.exists():
        print("❌ Error: ryan-dash directory not found!")
        return
    
    npm_path = find_npm()
    if not npm_path:
        print("❌ Error: npm not found! Please install Node.js and npm.")
        print("📥 Download from: https://nodejs.org/")
        return
    
    print("\n📦 Installing backend dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend_requirements.txt"], check=True)
        print("✅ Backend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing backend dependencies: {e}")
        return
    
    print("\n🔧 Starting Python Backend Server...")
    try:
        backend_process = subprocess.Popen([sys.executable, "main.py", "--backend"])
        print("✅ Backend server started on http://localhost:5000")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return
    
    time.sleep(3)
    
    original_dir = os.getcwd()
    os.chdir(dash_path)
    
    if not Path("node_modules").exists():
        print("\n📦 Installing frontend dependencies...")
        try:
            if platform.system().lower() == 'windows':
                subprocess.run([npm_path, "install", "--force"], check=True, shell=True)
            else:
                subprocess.run([npm_path, "install", "--force"], check=True)
            print("✅ Frontend dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing frontend dependencies: {e}")
            backend_process.terminate()
            os.chdir(original_dir)
            return
    
    print("\n🌐 Starting Next.js Frontend...")
    print("="*50)
    print("📊 Dashboard will be available at http://localhost:3000")
    print("🔧 Backend API available at http://localhost:5000")
    print("🔄 Press Ctrl+C to stop both servers")
    print("="*50)
    
    try:
        if platform.system().lower() == 'windows':
            frontend_process = subprocess.Popen([npm_path, "run", "dev"], shell=True)
        else:
            frontend_process = subprocess.Popen([npm_path, "run", "dev"])
        
        os.chdir(original_dir)
        
        while True:
            time.sleep(1)
            
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
        
        print("\n👋 Dashboard stopped successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting frontend: {e}")
        backend_process.terminate()
        os.chdir(original_dir)
        print("💡 Try running manually: cd ryan-dash && npm run dev")


def run_training(config):
    print("\n" + "="*50)
    print("🏋️ Starting Training Process (Stable-Baselines3 PPO)")
    print("="*50)
    try:
        import inspect
        def filter_env_config(cfg):
            valid_args = inspect.signature(MtEnv.__init__).parameters
            return {k: v for k, v in cfg.items() if k in valid_args}

        seed = int(config.get('seed', 42))
        set_global_seeds(seed)

        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
        except Exception:
            device = 'cpu'

        algo_key = config.get('algorithm', 'sb3_ppo').lower()
        if algo_key not in ('sb3_ppo', 'ppo'):
            algo_key = 'sb3_ppo'
        config.setdefault(algo_key, {})

        env_config = config["env"]
        symbols = env_config.get('symbols', ['EURUSD'])
        timeframes = env_config.get('timeframes', [15])
        base_cache = f"feature_cache_{'_'.join(symbols)}_{'_'.join(map(str, timeframes))}.pkl"
        try:
            import glob
            base_no_ext, _ = os.path.splitext(base_cache)
            matches = glob.glob(f"{base_no_ext}_*.pkl")
        except Exception:
            matches = []

        if matches:
            print(f"✅ Found cached features: {os.path.basename(matches[0])}")
            print("🚀 Skipping feature engineering, using cached features directly!")
            config["env"]["use_cached_features"] = True
        else:
            print("\n🔬 Running feature engineering and caching features (one-time)...")
            pre_cache_env_config = config["env"].copy()
            pre_cache_env_config["use_cached_features"] = False
            MtEnv(**filter_env_config(pre_cache_env_config))
            print("✅ Feature engineering and caching complete.")
            config["env"]["use_cached_features"] = True

        from model.ppo import make_vec_envs, build_policy_kwargs
        from stable_baselines3 import PPO

        sb3_cfg = config.get(algo_key, {})
        n_envs = int(sb3_cfg.get('n_envs', 1))
        use_subproc = bool(sb3_cfg.get('use_subproc', False))
        env = make_vec_envs(config['env'], n_envs=n_envs, use_subproc=use_subproc)

        policy_kwargs = build_policy_kwargs(config.get('model', {}))

        total_timesteps = int(sb3_cfg.get('total_timesteps', 1_000_000))
        n_steps = int(sb3_cfg.get('n_steps', 2048))
        batch_size = int(sb3_cfg.get('batch_size', 256))
        n_epochs = int(sb3_cfg.get('n_epochs', 10))
        learning_rate = float(sb3_cfg.get('learning_rate', 3e-4))
        gamma = float(sb3_cfg.get('gamma', 0.99))
        gae_lambda = float(sb3_cfg.get('gae_lambda', 0.95))
        clip_range = float(sb3_cfg.get('clip_range', 0.2))
        ent_coef = float(sb3_cfg.get('ent_coef', 0.0))
        vf_coef = float(sb3_cfg.get('vf_coef', 0.5))
        max_grad_norm = float(sb3_cfg.get('max_grad_norm', 0.5))

        tensorboard_log = os.path.abspath('./tensorboard_logs')
        os.makedirs(tensorboard_log, exist_ok=True)

        print("\n🏃 Starting SB3 PPO training...")
        model = PPO(
            policy='MultiInputPolicy',
            env=env,
            verbose=1,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )

        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        save_name = sb3_cfg.get('model_path', 'ppo_transformer_mtsim_final')
        model.save(save_name)
        print(f"\n✅ Training completed. Model saved to {os.path.abspath(save_name + '.zip')}")
    except Exception as e:
        print("\n❌ Training Failed:")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ryan RL Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gui", action="store_true", 
                       help="Start GUI mode with Next.js dashboard")
    parser.add_argument("--train", action="store_true", 
                       help="Run training mode")
    parser.add_argument("--backtest", action="store_true", 
                       help="Run backtest mode")
    parser.add_argument("--backend", action="store_true", 
                       help="Start backend server for real-time metrics")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to main config file")
    
    args = parser.parse_args()
    
    try:
        if args.gui:
            start_gui_mode()
        elif args.backend:
            try:
                from backend_server import app
                print("\n" + "="*50)
                print("🚀 Starting RL Trading Backend Server")
                print("="*50)
                print("📊 API available at http://localhost:5000")
                print("🔄 Press Ctrl+C to stop the server")
                app.run(host='0.0.0.0', port=5000, debug=False)
            except ImportError as e:
                print(f"❌ Error importing backend server: {e}")
                print("💡 Make sure backend_server.py exists and dependencies are installed")
        elif args.backtest:
            run_backtest()
        elif args.train:
            config = load_config(args.config)
            run_training(config)
        else:
            config = load_config(args.config)
            run_training(config)
            
    except Exception as e:
        print("\n❌ Fatal Error:")
        print(f"Error: {str(e)}")
        sys.exit(1)
