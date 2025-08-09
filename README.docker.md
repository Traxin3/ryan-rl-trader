# Docker & Dev Container Guide

This project ships with Docker, Docker Compose, and a VS Code Dev Container for easy setup.

## Services
- backend: Python API and training/backtest entrypoints (`main.py`).
- frontend: Next.js dashboard (in `ryan-dash`).
- trainer: one-off training job.
- tensorboard: visualizes logs at :6006.

## Quick Start (CPU)
- Build and run dev stack:
  - Windows PowerShell: docker compose up --build
- Backend: http://localhost:5000
- Frontend: http://localhost:3000

## GPU Enablement
- Install NVIDIA Container Toolkit.
- Use the GPU override:
  - docker compose -f docker-compose.yml -f docker-compose.gpu.override.yml up --build

## One-off Training
- CPU: docker compose --profile train run --rm trainer
- GPU: docker compose -f docker-compose.yml -f docker-compose.gpu.override.yml --profile train run --rm trainer

## Dev Container (VS Code)
- Open Folder in Container.
- Post-create installs torch CPU or CUDA (auto-detect via `nvidia-smi`), Python reqs, and frontend deps.

## Images
- Backend: Dockerfile at repo root (installs torch CPU by default; override TORCH_INDEX_URL at build for CUDA).
- Frontend: `ryan-dash/Dockerfile`.

## Security Scans
- GitHub Actions runs Trivy filesystem and image scans; fails on HIGH/CRITICAL.

## Tips
- Mounts persist checkpoints and logs via named volumes.
- Adjust `config/config.yaml` for training/backtest params.
