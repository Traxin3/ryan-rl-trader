# Project Ryan (gym-mtsim, Reinvented)



This project is the result of years of relentless research, experimentation, and iteration in the field of Reinforcement Learning (RL) for financial trading.  

The environment is inspired by [AminHP/gym-mtsim](https://github.com/AminHP/gym-mtsim), but has been **heavily modified and expanded** to handle modern RL workflows, proper feature engineering, and real-world trading constraints.  

Meanwhile, everyone’s been off playing with LLMs and generating anime pictures, and I’ve been here crying over getting a model that doesn’t overfit, a sane environment, and features that actually make sense.



Honestly, I'm uploading this to GitHub because I need some of that sweet, sweet GitHub computer juice. But the real goal is to push this codebase toward **actual live deployment**. There are still many updates and improvements to come, but this is a major milestone, for me anyway.

## Technical Overview

### Environment
- **Custom RL Environment**: Built on top of `gymnasium`, inspired by `gym-mtsim` but with major enhancements for multi-symbol, multi-timeframe, and market-structure-aware trading.
- **Simulator**: Realistic order management, margin/leverage, and tick-by-tick simulation using historical data. Supports both local data caching and cloud/offline workflows with detailed tqdm progress bars for both loading and downloading data.
- **Feature Engineering**: 
  - Robust, compressed, and cached feature extraction pipeline.
  - Uses `VarianceThreshold`, `StandardScaler`, and `PCA` to reduce 30k+ raw features to a compact latent space.
  - Market structure, liquidity, risk, and temporal features are all included.
  - Feature cache is validated for shape and parameters to avoid unnecessary recomputation.

### Model
- **Policy**: Transformer-based feature extractor (`TransformerFeatureExtractor`) for handling temporal dependencies in price/feature windows.
- **RL Algorithm**: Proximal Policy Optimization (PPO) from `stable-baselines3`.
- **Custom Policy**: `TransformerPolicy` with separate actor/critic networks and a transformer encoder for feature extraction.
- **Training**: Supports vectorized environments, TensorBoard logging, and custom risk metrics (win rate, Sharpe, Sortino, drawdown, etc.).
- **Evaluation**: Includes detailed evaluation loop and advanced rendering for visualizing agent performance.

### Data Handling
- **Symbols & Timeframes**: Easily configurable for any symbol/timeframe combination.
- **Local Caching**: All market data is cached locally for fast reloads; tqdm progress bars show both loading and downloading status.
- **Offline/Cloud Ready**: Can fetch data from MetaTrader 5 or load from local cache, making it suitable for both research and deployment.

### Logging & Metrics
- **TensorBoard**: Logs all key metrics, including custom risk metrics, for easy experiment tracking. 
- **Custom Callbacks**: `TensorboardMetricsCallback` tracks win rate, average profit, drawdown, Sharpe/Sortino ratios, and more. -> Uhm not sure if this even works

## How to Use

1. **Install requirements** (see `requirements.txt`).
2. **Configure your environment** in `main.py` (symbols, timeframes, window size, etc.).
3. **Run training**:
   ```bash
   python main.py
   ```
4. **Monitor progress** in TensorBoard:
   ```bash
   tensorboard --logdir=./tensorboard_logs/
   ```
5. **Evaluate and visualize** results after training.

## Roadmap
- [ ] Live trading integration (MetaTrader, broker APIs)
- [ ] More advanced risk management and position sizing
- [ ] Hyperparameter search and experiment tracking
- [ ] More robust backtesting and slippage modeling
- [ ] Improved documentation and tutorials

## Final Thoughts
This project is a living, breathing RL research lab for trading. It's the product of endless curiosity, trial and error, and a refusal to give up. If you're on a similar journey, welcome aboard. Contributions, feedback, and PRs are always welcome!

Ill leave a quote down to sound cool.

---

*“The markets are never wrong, only opinions are.”* — Jesse Livermore
