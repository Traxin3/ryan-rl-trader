# 🚀 Project Ryan – (gym-mtsim, Reinvented, again)

[![Project Status](https://img.shields.io/badge/Status-🚧_In_Development-orange?style=for-the-badge)]()  
[![Made with Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)]()  
[![RL Stack](https://img.shields.io/badge/Stable--Baselines3-PPO-3C91E6?style=for-the-badge)]()  
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=24&duration=4000&pause=1000&color=00FFD1&center=true&vCenter=true&width=700&lines=🤖+Reinforcement+Learning+Trading;🧠+Stable-Baselines3+PPO+%2B+Transformer;📊+Backtesting+%26+Research" />
</p>

---

## ✨ Highlights

- Stable-Baselines3 PPO with a custom Transformer features extractor
- Multi-timeframe analysis (1m, 5m, 15m, 1h, ...)
- Dict observations with liquidity and orders fusion
- Deterministic feature cache for faster training
- Risk controls: dynamic SL/TP and position sizing
- VectorBT backtesting with rich analytics

> Note: The dashboard (ryan-dash) is currently inactive and removed from this README.

---

## 🆕 What’s New

- Migrated off Ray/RLlib → standardized on Stable-Baselines3 PPO
- New SB3 Transformer extractor that wraps the internal TransformerFeatureExtractor
- Safer sequence handling (pad/trim) and fusion of liquidity/orders branches
- Config-driven training via `config/config.yaml` (algorithm: `sb3_ppo`)
- Vectorized envs (Windows-safe) and cleaner training pipeline (no callbacks)
- Backtesting continues to use the saved SB3 model

---

## 🚀 Quick Start

1) Install dependencies
```bash
git clone https://github.com/Traxin3/ryan-rl-trader
cd ryan-rl-trader
pip install -r requirements.txt
```

2) Train
```bash
python main.py --train
```
- Saves the model to the path configured in `config/config.yaml`
- TensorBoard logs in `tensorboard_logs/`

3) Backtest
```bash
python main.py --backtest
```
- Generates analytics and reports (see backtest module)

4) Optional: View TensorBoard
```bash
tensorboard --logdir tensorboard_logs
```

---

## ⚙️ Configuration (YAML)

`config/config.yaml`
```yaml
algorithm: sb3_ppo

sb3_ppo:
  total_timesteps: 2000000
  n_envs: 4
  use_subproc: false
  n_steps: 2048
  batch_size: 512
  n_epochs: 10
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  model_path: "ppo_transformer_mtsim_final.zip"

model:
  transformer:
    d_model: 256
    nhead: 8
    num_layers: 4
    dropout: 0.1
    scales: [1, 2, 4]

env:
  symbol: "EURUSD"
  timeframe: 15
  window_size: 256
```

---

## 🏗 Architecture (Brief)

- MtEnv + MtSimulator (execution, market impact, orders)
- Transformer-based features extractor (with liquidity/orders fusion)
- Stable-Baselines3 PPO (MultiInput policy)
- VectorBT backtesting and reporting

---

## 🧪 Troubleshooting

- Ensure dependency pairing: `stable-baselines3==2.0.0` with `gymnasium==0.28.1`
- CUDA issues: verify your PyTorch build matches your CUDA toolkit
- OOM: reduce `batch_size`, `n_steps`, or `window_size`

---

## 🗺️ WIP / Roadmap

Heavily working on the model and training recipes. Still need to:
- Clean up code and modules
- Update functionality and configs
- Enhance pipelines and data flow
- Expand tests and benchmarks

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a PR

---

## 📣 Shoutouts / Contact

- Discord: `seany519`

---

## 📜 License

MIT License – see [LICENSE](LICENSE)
