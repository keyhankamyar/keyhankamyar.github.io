---
title: "Projects"
url: "/projects/"
ShowReadingTime: false
ShowBreadCrumbs: true
ShowPostNavLinks: false
ShowWordCount: false
---

## 🚀 Shipped

### SpaX
[![GitHub](https://img.shields.io/badge/GitHub-SpaX-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/SpaX)
[![PyPI](https://img.shields.io/pypi/v/spax?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/spax/)
[![Python](https://img.shields.io/badge/Python-3.11--3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

Pythonic, type-safe search space configuration for hyperparameter optimization, neural architecture search, and ML experiment tracking. Built to eliminate boilerplate and enforce best practices from research to production.

**Features:**
- Declarative search space definition with automatic inference
- Conditional parameters with complex dependency logic
- Nested and polymorphic configurations
- Native Optuna integration for HPO
- Iterative search space refinement based on results
- Multi-format serialization (JSON/YAML/TOML)

**Stack:** Python • Pydantic • Optuna

---

### TickVault
[![GitHub](https://img.shields.io/badge/GitHub-TickVault-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/TickVault)
[![PyPI](https://img.shields.io/pypi/v/tick-vault?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/tick-vault/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

High-performance financial tick data pipeline for Dukascopy Bank's historical datafeed. Built for quantitative researchers and algorithmic traders who need reliable access to high-resolution market data.

**Features:**
- Concurrent downloads with intelligent resume capability
- Multi-proxy pipeline for distributed downloading
- Efficient decompression and decoding
- SQLite metadata tracking and gap detection
- Pandas and NumPy integration

**Stack:** Python • httpx • NumPy • Pandas • SQLite • LZMA

---

### ProxyRotator
[![GitHub](https://img.shields.io/badge/GitHub-ProxyRotator-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/ProxyRotator)
[![PyPI](https://img.shields.io/pypi/v/xray-proxy-rotator?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/xray-proxy-rotator/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

Async Python library for managing VMESS proxy and user-agent rotation with automatic subscription updates, connection testing, and stealth-focused user-agent selection. Built for resilient web scraping workflows.

**Features:**
- Automatic proxy rotation with subscription support
- Connection testing and filtering of working proxies
- User-agent rotation with globally popular patterns
- Rate limiting with jitter for natural request patterns
- Thread-safe with context manager support

**Stack:** Python • httpx • Xray-core • Pydantic • asyncio

---

## 🔬 In Development

### Clean-TS
Modular, Pythonic reimplementation of canonical time-series architectures. Makes archaic, opaque TS codebases readable, extensible, and reproducible.

**Status:** Refactoring • ~1 month to release

---

### Lightning HPO Playbooks
Industry-standard examples and guides for model training, optimization, and research using PyTorch Lightning. Covers SOTA practices for NAS, HPO, distributed training, and production-ready ML pipelines.

**Status:** Planning

---

### Financial RL Environment
High-performance, parallelized Gymnasium environment for algorithmic trading research. Built for large-scale RL training with custom reward formulations.

**Status:** Planning