---
title: "Projects"
url: "/projects/"
ShowReadingTime: false
ShowBreadCrumbs: true
ShowPostNavLinks: false
ShowWordCount: false
---

## 🔓 Open Source

### TickVault
[![GitHub](https://img.shields.io/badge/GitHub-TickVault-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/TickVault)
[![PyPI](https://img.shields.io/pypi/v/tick-vault?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/tick-vault/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

High-performance financial tick data pipeline for Dukascopy Bank's historical datafeed. Built for quantitative researchers and algorithmic traders who need reliable access to high-resolution market data.

**Features:**
- Concurrent downloads with intelligent resume capability
- Multi-Proxy pipeline for distributed downloading
- Efficient decompression and decoding
- SQLite metadata tracking and gap detection
- Pandas and NumPy integration

**Stack:** Python • httpx • NumPy • Pandas • SQLite • LZMA

---

### ProxyRotator
[![GitHub](https://img.shields.io/badge/GitHub-ProxyRotator-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/ProxyRotator)
[![PyPI](https://img.shields.io/pypi/v/xray-proxy-rotator?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/xray-proxy-rotator/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

An async Python library built for **stealth** for managing VMESS proxy and user-agent rotation with automatic subscription updates, connection testing, and automatic selection of most used user-agents for maximum anonymity. Perfect for web scraping projects requiring reliable proxy management.

**Features:**
- Automatic proxy rotation with subscription support
- Connection testing and filtering of working proxies
- User-agent rotation for each proxy
- Uses most used user-agents globally for maximum stealth
- Rate limiting with jitter for natural request patterns
- Thread-safe with context manager support

**Stack:** Python • httpx • Xray-core • Pydantic • asyncio

---

## 🔬 Coming Soon

### SAM2 Realtime Predictor
Real-time video segmentation optimized for live camera feeds and streaming video. Text-prompted segmentation, custom kernel optimizations and TensorRT compilation.

**Status:** Polishing • Ready <2 weeks

---

### Tunable Config
Research-grade library unifying Optuna (HPO), neural architecture search, and feature selection with type-safe Pydantic configs. Distilled from years of experimentation into clean, reusable patterns.

**Status:** Planning • Expected in November

---

### Clean-TS
Modular, Pythonic reimplementation of canonical time-series architectures. Makes archaic, opaque TS codebases readable, extensible, and reproducible.

**Status:** Refactoring