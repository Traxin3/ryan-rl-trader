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
from model.ppo_model import create_ppo_model
from gym_mtsim.envs.mt_env import MtEnv
from stable_baselines3.common.env_util import make_vec_env
import warnings

warnings.filterwarnings('ignore')

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

def create_env(config, test_mode: bool = False):
    """Create trading environment from config with optional test mode settings"""
    env_config = config['env']
    
    if test_mode:
        return MtEnv(
            symbols=env_config['symbols'],
            timeframes=env_config['timeframes'],
            window_size=env_config['window_size'],
            max_leverage=env_config['max_leverage'],
            reward_scaling=1.0,  # Disable reward scaling for backtests
            risk_adjusted_reward=False  # Disable for cleaner backtest results
        )
    
    return MtEnv(
        symbols=env_config['symbols'],
        timeframes=env_config['timeframes'],
        window_size=env_config['window_size'],
        max_leverage=env_config['max_leverage'],
        reward_scaling=env_config.get('reward_scaling', 1.0),
        risk_adjusted_reward=env_config.get('risk_adjusted_reward', True)
    )

def run_backtest():
    """Execute backtest using configured parameters"""
    print("\n" + "="*50)
    print("🚀 Starting Backtest")
    print("="*50)
    
    try:
        config = load_config()
        backtest_config = load_backtest_config()['backtest']
        
        env = create_env(config, test_mode=True)
        
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
        backtester.generate_report(report_path)
        
        duration = time.time() - start_time
        print("\n" + "="*50)
        print("✅ Backtest Completed Successfully")
        print(f"⏱️ Duration: {duration:.2f} seconds")
        print("="*50)
        print("\n📊 Performance Summary:")
        print(f"  Total Return: {results['total_return']:.2%}")
        print(f"  Annualized Return: {results['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"  Win Rate: {results['win_rate']:.2%}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Total Trades: {results['total_trades']}")
        print("\n📂 Results saved to:")
        print(f"  {os.path.abspath(report_path)}")
        print(f"  {os.path.abspath('backtest_results/performance.html')}")
        print("="*50)
        
    except Exception as e:
        print("\n❌ Backtest Failed:")
        print(f"Error: {str(e)}")
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
    """Run the training process with enhanced logging"""
    print("\n" + "="*50)
    print("🏋️ Starting Training Process")
    print("="*50)
    
    try:
        venv = make_vec_env(lambda: create_env(config), n_envs=1)
        model = create_ppo_model(venv, config)
        
        print("\n🔧 Model Architecture:")
        print(model.policy)
        
        print("\n🏃 Starting training...")
        start_time = time.time()
        model.learn(total_timesteps=1_500_000, progress_bar=True)
        
        model_path = "ppo_transformer_mtsim_final"
        model.save(model_path)
        
        duration = time.time() - start_time
        print("\n" + "="*50)
        print(f"✅ Training Completed in {duration/60:.2f} minutes")
        print(f"💾 Model saved to: {model_path}")
        print("="*50)
        
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
