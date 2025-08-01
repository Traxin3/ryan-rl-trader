# Project Ryan (gym-mtsim, Reinvented, again) - this time with GUI Dashboard

A comprehensive Reinforcement Learning trading system with a futuristic Next.js dashboard for real-time monitoring and control.

## Features

### 🚀 Core RL System
- **PPO Algorithm**: Proximal Policy Optimization with Transformer architecture, but will soon be updated to support Multi-Agents(Yeah soon)
- **Multi-timeframe Analysis**: Support for multiple timeframes (1m, 5m, 15m, 1h, etc.)
- **Advanced Feature Engineering**: 100+ core features with PCA optimization
- **Risk Management**: Dynamic stop-loss, take-profit, and position sizing
- **Market Structure Analysis**: Liquidity based features

### 🎛️ GUI Dashboard
- **Real-time Monitoring**: Live metrics, performance charts, and system status
- **Training Control**: Start/stop training, save models, export data
- **Configuration Management**: Edit model parameters, environment settings
- **History Tracking**: Compare training runs, view performance metrics
- **System Monitoring**: GPU/CPU usage, memory monitoring
- **Futuristic UI**: Animated components, smooth transitions, modern design

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
git clone https://github.com/Traxin3/ryan-rl-trader


```

### 2. Run in GUI Mode

```bash
# Start the almost cool looking dashboard
python main.py --gui
```

This will:
- Start the Next.js development server on `http://localhost:3000`
- Open the dashboard in your browser
- Provide real-time control over the RL system

### 3. Run Training Mode

```bash
# Run training without GUI(lame mode)
python main.py --train

# Or run with default settings(simple life mode)
python main.py
```

## Dashboard Features

### 📊 Overview Tab
- **Real-time Metrics**: Balance, equity, win rate, Sharpe ratio
- **Control Panel**: Start/stop training, save models, export data
- **Performance Charts**: Live portfolio performance visualization
- **System Status**: GPU/CPU usage, memory monitoring

### 🎯 Training Tab
- **Training Status**: Real-time progress, steps, episodes
- **Model Configuration**: Edit PPO hyperparameters
- **Live Metrics**: Current reward, win rate, performance indicators

### 📈 Metrics Tab
- **Advanced Metrics**: Sortino ratio, profit factor, volatility
- **Trade Analysis**: Average trade profit, holding times
- **Risk Metrics**: Maximum drawdown, risk-adjusted returns

### ⚙️ Configuration Tab
- **Environment Settings**: Symbols, timeframes, window size
- **Model Parameters**: Learning rate, batch size, epochs
- **Risk Settings**: Max leverage, reward scaling

### 📚 History Tab
- **Training History**: View past training runs
- **Performance Comparison**: Compare different model versions
- **Export Options**: Download models and data

### 🖥️ System Tab
- **Resource Monitoring**: GPU, CPU, memory usage
- **System Status**: Environment connectivity, data feed status
- **Logging**: Real-time system logs

## Configuration

The system uses `config/config.yaml` for all settings:

```yaml
# PPO Hyperparameters
ppo:
  learning_rate: 0.0004
  n_steps: 2048
  batch_size: 256
  n_epochs: 5
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  max_grad_norm: 0.5
  vf_coef: 0.25

# Environment
env:
  symbols: ["EURUSD"]
  timeframes: [15]
  window_size: 50
  max_leverage: 2.0
  reward_scaling: 1.0
  risk_adjusted_reward: true
```

## Architecture

### RL Components
- **MtEnv**: Custom Gym environment for MetaTrader simulation
- **TransformerPolicy**: PPO with Transformer feature extractor
- **MtSimulator**: Advanced trading simulator with market structure analysis
- **FeatureEngine**: Optimized feature extraction with caching

### Dashboard Components
- **Next.js App**: Modern React framework with server-side rendering
- **Framer Motion**: Smooth animations and transitions
- **Recharts**: Interactive data visualization
- **Lucide React**: Beautiful icon library
- **Tailwind CSS**: Utility-first styling

### API Integration
- **Training API**: Control training processes
- **Config API**: Manage configuration files
- **Metrics API**: Real-time performance data
- **System API**: Resource monitoring

## Development

### Adding New Features

1. **RL System**: Modify `gym_mtsim/envs/mt_env.py` for new environment features
2. **Model**: Update `model/transformer.py` for new architectures
3. **Dashboard**: Add new components in `ryan-dash/app/`
4. **API**: Create new routes in `ryan-dash/app/api/`

### Customization

- **UI Theme**: Modify `ryan-dash/app/globals.css` for styling
- **Metrics**: Add new metrics in the dashboard components
- **Charts**: Integrate additional Recharts components
- **Animations**: Use Framer Motion for custom animations

## Performance

### Optimization Features
- **Feature Caching**: Automatic caching of computed features
- **PCA Optimization**: Dimensionality reduction for efficiency
- **Multi-processing**: Parallel feature computation
- **Memory Management**: Efficient data structures and cleanup

### Monitoring
- **Real-time Metrics**: Live performance tracking
- **Resource Usage**: GPU/CPU/memory monitoring
- **Training Progress**: Step-by-step progress tracking
- **Error Handling**: Comprehensive error reporting

## Troubleshooting

### Common Issues

1. **Dashboard not starting**: Check Node.js installation and dependencies
2. **Training errors**: Verify Python dependencies and CUDA installation
3. **Config issues**: Ensure `config/config.yaml` exists and is valid
4. **Memory issues**: Reduce batch size or window size

### Debug Mode

```bash
# Run with verbose logging
python main.py --train --verbose

# Check dashboard logs
cd ryan-dash && npm run dev
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🚀 Ready to trade with AI? Start the GUI and begin your RL trading journey!**
