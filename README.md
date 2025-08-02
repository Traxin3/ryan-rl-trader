# 🚀 Project Ryan – (gym-mtsim, Reinvented, again)

[![Project Status](https://img.shields.io/badge/Status-🚧_In_Development-orange?style=for-the-badge)]()  
[![Made with Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)]()  
[![Dashboard](https://img.shields.io/badge/Next.js-Futuristic_Dashboard-black?style=for-the-badge&logo=next.js)]()  
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=24&duration=4000&pause=1000&color=00FFD1&center=true&vCenter=true&width=650&lines=🤖+Reinforcement+Learning+Trading;🎛️+Real-Time+Next.js+Dashboard;📊+Now+with+Professional+Backtesting">
</p>

---

## ✨ Features

### 🚀 Core RL System
- **PPO Algorithm** → Transformer-powered PPO (Multi-Agent coming soon)
- **Multi-timeframe Analysis** → 1m, 5m, 15m, 1h, etc.
- **Feature Engineering** → 100+ features + PCA optimization
- **Risk Management** → Dynamic SL/TP + position sizing
- **Market Structure Analysis** → Liquidity-based signals

### 🎛️ GUI Dashboard
- **Live Monitoring** → Balance, equity, Sharpe ratio, and system metrics
- **Training Control** → Start/stop, save models, export data
- **Config Management** → Edit model/env params live
- **History Tracking** → Compare training runs & performance
- **System Monitoring** → GPU/CPU/memory live
- **Smooth UI** → Subtle animations, clean layout

### 📊 New: VectorBT Backtesting
- **Professional Analytics**: Sharpe, Sortino, max drawdown metrics
- **Interactive Reports**: HTML reports with trade-by-trade analysis
- **Flexible Configuration**: Test specific symbols, timeframes, and periods
- **Visualization**: Equity curves, underwater plots, and trade timelines

---

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=20&duration=3000&pause=1000&color=FFB800&center=true&vCenter=true&width=600&lines=🚀+Quick+Start">
</p>

### 1️⃣ Install Dependencies
```bash
git clone https://github.com/Traxin3/ryan-rl-trader
cd ryan-rl-trader
pip install -r requirements.txt
```

### 2️⃣ Run Modes
#### GUI Mode (Recommended)
```bash
python main.py --gui
```
- Starts dashboard at http://localhost:3000
- Real-time monitoring and control
- Visualize training and backtest results

#### Training Mode
```bash
python main.py --train
```
- Headless training session
- Saves model to `ppo_transformer_mtsim_final`
- TensorBoard logging available

#### Backtest Mode
```bash
python main.py --backtest
```
- Runs backtest with default settings
- Generates comprehensive HTML report
- Saves results to `backtest_results/`

#### Combined Mode
```bash
python main.py --gui --train --backtest
```
- Runs all components together
- Dashboard shows live training and backtest results

### 📊 Dashboard Tabs
- ✅ **Overview** – Metrics, GPU/CPU, charts
- 🎯 **Training** – Progress, hyperparameters, rewards
- 📈 **Backtest (NEW)** – Interactive HTML reports, strategy comparisons
- ⚙️ **Config** – Symbols, timeframes, risk settings
- 📚 **History** – Training runs, performance comparisons
- 🖥 **System** – Logs & resource usage

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=20&duration=3000&pause=1000&color=5BFF7F&center=true&vCenter=true&width=650&lines=⚙️+Configuration+via+YAML">
</p>

### 🔧 Main Config (`config/config.yaml`)
```yaml
ppo:
  learning_rate: 0.0004
  n_steps: 2048
  batch_size: 256

env:
  symbols: ["EURUSD"]
  timeframes: [15]
  max_leverage: 2.0
```

### 📈 Backtest Config (`config/backtest_config.yaml`)
```yaml
backtest:
  model_path: "ppo_transformer_mtsim_final"
  test_start: "2023-01-01"
  test_end: "2023-06-30"
  output:
    report_path: "backtest_results/report.html"
```

### 🛠 Command Line Options
| Option        | Description                  | Example                         |
|---------------|------------------------------|---------------------------------|
| `--gui`       | Start the dashboard          | `python main.py --gui`         |
| `--train`     | Run training session         | `python main.py --train`       |
| `--backtest`  | Run backtest                | `python main.py --backtest`    |
| `--test_start`| Backtest start date         | `--test_start "2023-01-01"`   |
| `--test_end`  | Backtest end date           | `--test_end "2023-06-30"`     |
| `--symbol`    | Override symbol             | `--symbol EURUSD`             |
| `--timeframe` | Override timeframe          | `--timeframe 5`               |

---

## 🏗 Architecture
### RL Core
- **MtEnv** → Custom Gym environment for MetaTrader sim
- **TransformerPolicy** → PPO + Transformer
- **MtSimulator** → Advanced trading simulator
- **FeatureEngine** → Feature caching + PCA

### Dashboard Core
- **Next.js + Tailwind** → Clean futuristic UI
- **Framer Motion** → Subtle transitions
- **Recharts** → Live metrics
- **Lucide React** → Icon library

### Backtesting Core
- **VectorBT Integration** → Professional-grade backtesting
- **Trade Analytics** → Sharpe, Sortino, drawdowns
- **HTML Reports** → Interactive trade exploration

---

## 🛠 Troubleshooting
- **Dashboard won’t start?** → Check Node.js install
- **Training errors?** → Validate Python deps & CUDA
- **Memory issues?** → Lower `batch_size` or `window_size`
- Run debug mode:
```bash
python main.py --train --verbose
cd ryan-dash && npm run dev
```

---

## 🤝 Contributing
1. Fork repo
2. Create feature branch
3. Make changes
4. Open PR 🚀

---

## 📜 License
MIT License – see [LICENSE](LICENSE)

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=22&duration=3000&pause=1000&color=FF00A2&center=true&vCenter=true&width=700&lines=🚀+Ready+to+Trade+with+AI%3F;🎛️+Fire+up+the+Dashboard+and+Backtest+Strategies!">
</p>