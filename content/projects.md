---
title: "Projects"
url: "/projects/"
ShowReadingTime: false
ShowBreadCrumbs: true
ShowPostNavLinks: false
ShowWordCount: false
---

## 🚀 Open Source

### TickVault
[![GitHub](https://img.shields.io/badge/GitHub-TickVault-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/TickVault)
[![PyPI](https://img.shields.io/pypi/v/tick-vault?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/tick-vault/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

High-performance financial tick data scraper for Dukascopy Bank's historical datafeed. Built for quantitative researchers and algorithmic traders who need reliable access to high-resolution market data.

**Features:**
- Concurrent downloads with intelligent resume capability
- Proxy support for distributed downloading
- Efficient LZMA decompression and NumPy-based decoding
- Pandas integration with gap detection
- SQLite metadata tracking

**Stack:** Python • httpx • NumPy • Pandas • SQLite • LZMA

---

### ProxyRotator
[![GitHub](https://img.shields.io/badge/GitHub-ProxyRotator-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/ProxyRotator)
[![PyPI](https://img.shields.io/pypi/v/xray-proxy-rotator?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/xray-proxy-rotator/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

Production-ready async Python library for managing VMESS proxy rotation with automatic subscription updates, connection testing, and user-agent rotation. Perfect for web scraping projects requiring reliable proxy management.

**Features:**
- Automatic proxy rotation with subscription support
- Connection testing and filtering of working proxies
- User-agent rotation for each proxy
- Rate limiting with jitter for natural request patterns
- Thread-safe with context manager support

**Stack:** Python • httpx • Xray-core • Pydantic • asyncio

---

## 🔬 Research & Tools (Coming Soon)

### SAM2 Realtime Predictor
Real-time video segmentation optimized for live camera feeds and streaming video. Text-prompted segmentation achieving 40+ FPS through custom kernel optimizations and TensorRT compilation.

**Status:** Polishing • Expected Q1 2025

---

### Tunable Config
Research-grade library unifying Optuna (HPO), neural architecture search, and feature selection with type-safe Pydantic configs. Distilled from years of experimentation into clean, reusable patterns.

**Status:** Planning • Expected Q2 2025

---

### Clean-TS
Modular, Pythonic reimplementation of canonical time-series architectures. Makes archaic, opaque TS codebases readable, extensible, and reproducible.

**Status:** Refactoring • Expected Q2 2025

---

## 💼 Production Work

Throughout my career, I've built production systems that aren't open-source but demonstrate real-world ML engineering:

**Sentiment Analysis Pipeline** — MetaScape, 2022-2024  
News scraping, SBERT embeddings, financial sentiment classification with 69% accuracy. Handled automatic retraining and drift detection in production.

**Synthetic Data Generation** — Dideo, 2020-2021  
Human-in-the-loop system for nudity detection. Achieved 200x speedup in data generation through active learning and automatic nightly finetuning.

**RL Trading Framework** — Independent, 2021-Present  
Custom Gymnasium environments, distributed training, HPO + NAS pipelines. Multi-armed bandit reformulation of trading problems. Currently training at scale (compute-limited).

**Defect Detection** — Contract Work, 2022  
YOLOv7-based quality control for pottery manufacturing. Reduced supervision staff from 5 to 1 with just a PC and webcam.

