import os
import yaml
import argparse
import subprocess
import sys
import json
import time
import threading
from pathlib import Path
from model.ppo_model import create_ppo_model
from gym_mtsim.envs.mt_env import MtEnv
from stable_baselines3.common.env_util import make_vec_env

def load_config():
    config_paths = [
        "config/config.yaml",  # Local development path
        "/content/ryan-rl-trader/config/config.yaml",  # Google Colab path
        os.path.join(os.path.dirname(__file__), "config/config.yaml")  # Relative to script
    ]
    
    for path in config_paths:
        try:
            with open(path) as f:
                print(f"Successfully loaded config from: {path}")
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError(
        f"Could not find config.yaml in any of these locations:\n"
        f"{config_paths}\n"
        f"Please ensure the config file exists in one of these paths."
    )

def create_env(config):
    return MtEnv(
        symbols=config['env']['symbols'],
        timeframes=config['env']['timeframes'],
        window_size=config['env']['window_size'],
        max_leverage=config['env']['max_leverage'],
        reward_scaling=config['env'].get('reward_scaling', 1.0),
        risk_adjusted_reward=config['env'].get('risk_adjusted_reward', True)
    )

def find_npm():
    """Find npm executable in common locations"""
    npm_paths = [
        "npm",  # Try direct npm command
        "C:\\Program Files\\nodejs\\npm.cmd",  # Windows default
        "C:\\Program Files (x86)\\nodejs\\npm.cmd",  # Windows 32-bit
        os.path.expanduser("~\\AppData\\Roaming\\npm\\npm.cmd"),  # User npm
        os.path.expanduser("~\\AppData\\Local\\npm\\npm.cmd"),  # Local npm
    ]
    
    for path in npm_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Found npm at: {path}")
                return path
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    
    return None

def start_gui_mode():
    """Start the complete RL Trading Dashboard with backend and frontend"""
    print("🚀 Starting RL Trading Dashboard...")
    print("=" * 50)
    
    dash_path = Path("ryan-dash")
    if not dash_path.exists():
        print("❌ Error: ryan-dash directory not found!")
        return
    
    npm_path = find_npm()
    if not npm_path:
        print("❌ Error: npm not found! Please install Node.js and npm.")
        print("📥 Download from: https://nodejs.org/")
        return
    
    print("📦 Installing backend dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend_requirements.txt"], check=True)
        print("✅ Backend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing backend dependencies: {e}")
        return
    
    print("🔧 Starting Python Backend Server...")
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
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run([npm_path, "install", "--force"], check=True)
            print("✅ Frontend dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing frontend dependencies: {e}")
            backend_process.terminate()
            os.chdir(original_dir)
            return
    
    print("🌐 Starting Next.js Frontend...")
    print("📊 Dashboard will be available at http://localhost:3000")
    print("🔧 Backend API available at http://localhost:5000")
    print("🔄 Press Ctrl+C to stop both servers")
    print("=" * 50)
    
    try:
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
        
        print("👋 Dashboard stopped successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting frontend: {e}")
        backend_process.terminate()
        os.chdir(original_dir)
        print("💡 Try running manually: cd ryan-dash && npm run dev")

def run_training(config):
    """Run the training process"""
    try:
        venv = make_vec_env(lambda: create_env(config), n_envs=1)
        model = create_ppo_model(venv, config)
        
        model.learn(total_timesteps=1_500_000, progress_bar=True)
        model.save("ppo_transformer_mtsim_final")
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ryan RL Trading System")
    parser.add_argument("--gui", action="store_true", help="Start GUI mode with Next.js dashboard")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--backend", action="store_true", help="Start backend server for real-time metrics")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    if args.gui:
        start_gui_mode()
    elif args.backend:
        try:
            from backend_server import app
            print("🚀 Starting RL Trading Backend Server...")
            print("📊 API available at http://localhost:5000")
            print("🔄 Press Ctrl+C to stop the server")
            app.run(host='0.0.0.0', port=5000, debug=False)
        except ImportError as e:
            print(f"❌ Error importing backend server: {e}")
            print("💡 Make sure backend_server.py exists and dependencies are installed")
    elif args.train:
        config = load_config()
        run_training(config)
    else:
        config = load_config()
        run_training(config)
