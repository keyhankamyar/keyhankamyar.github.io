---
title: "I Downloaded 2,000,000+ Hours of Market Tick Data So You Don't Have To"
date: 2025-10-15T12:00:00+03:30
draft: false
tags: ["machine-learning", "finance", "open-source", "python", "data-engineering"]
categories: ["Projects"]
description: "TickVault: Open-source Python library for downloading high-quality financial tick data from Dukascopy. Built for quantitative researchers and algorithmic traders. Handles parallel downloads, rate limits, resume capability, and on-demand decoding. Free alternative to expensive tick data APIs."
images: 
  - /images/tickvault-og.png

keywords: 
  - financial tick data
  - tick data download
  - dukascopy python library
  - dukascopy tick data
  - forex tick data
  - historical market data
  - algorithmic trading data
  - quantitative finance tools
  - backtesting framework
  - high frequency trading data
  - reinforcement learning finance
  - machine learning trading
  - python trading library
  - market data pipeline
  - financial data engineering
  - tick data analysis
  - real-time market data
  - trading system development
  - quant research tools
  - open source trading tools

author: "Keyhan Kamyar"
canonicalURL: "https://keyhankamyar.github.io/posts/tickvault-introduction/"
---

If you've ever tried to work with financial market data at scale, you know the pain points - and you're in the right place. If you're just starting, great timing. This post can save you months of frustration.

We'll discuss the **curse of resampling**, why **existing tools break at scale**, and how to actually solve these problems.

[![GitHub](https://img.shields.io/badge/GitHub-TickVault-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/TickVault)
[![PyPI](https://img.shields.io/pypi/v/tick-vault?style=flat-square)](https://pypi.org/project/tick-vault/)

---

### ⚠️ The Problem

Let me start with the quick overview, then we'll dig into the details.

**Arbitrary resampling** hides the movements you actually care about. Your model says enter at $1,850.23, exit at $1,850.89—but your hourly candle shows High: $1,851.20, Low: $1,849.80. Did your stop-loss trigger first, or did you hit take-profit? *The data doesn't tell you.*

**Tools that break at scale.** Libraries that crash on large downloads, can't resume, take forever to decode, or try to load everything into memory. Half are abandoned, the other half weren't built for production workloads.

**Rate limits** turn a 3-day download into a 3-week ordeal. Dukascopy's free datafeed is excellent—until you need 30 symbols across 20 years and suddenly find yourself bottlenecked at 5 requests per second.

**Expensive APIs** for data that should be free. Why pay hundreds per month when the raw data exists—if only there was a proper way to access it?

Over the past few months, I downloaded and decoded **2,000,000+ hours** of tick data from Dukascopy. Not because I enjoy building tools*—because I needed it for reinforcement learning research. So I built **TickVault**.

*Okay, I do enjoy building tools. But this one was pure necessity.

---

## 🚩 Why Existing Solutions Fall Short

### Why Tick Data? The Resampling Trap

Most data sources give you pre-resampled data—hourly candles, 5-minute bars, daily OHLC. But who decided that's the right granularity for *your* problem?

**Assumptions are the worst enemy of Machine Learning models.** Here's why:

Machine Learning models are **"function approximators"**. They will see what a function inputs, and what it outputs and try their best to mimic the behavior of that function that would results into the outcome. You wouldn't expect a function to receive only a subset of the inputs it needs and still give you the same output, would you? So we need to give all of the required inputs to a model, then hope that model learns to output the correct label.

Raw tick data has inherent challenges:

- **Too granular** → Massive dimensions, heavy memory and compute requirements
- **Irregular timing** → Doesn't follow structured patterns in time

The traditional solution? **Resampling.** Bucket the data into fixed time bins and handpick representative values. More manageable for humans, but riddled with problems:

**The first problem: Hidden assumptions.**

**❌ Event Loss:** Fast events compressed into one bucket lose critical detail. Multi-timeframe charts tried to help, but added complexity.

**❌ Hidden Patterns:** Arbitrary bin widths (1-hour, 5-minute) can mask cyclical patterns your model needs to see.

**❌ Feature Limitation:** OHLC (Open, High, Low, Close) throws away every tick between those four points—and the *order* of events.

If you're training an ML model, why would you assume 1-hour bins are the optimal feature resolution? And even if hourly happens to work, why assume that Open, High, Low, Close captures what matters? You're throwing away every tick between those four points and the order of events.

**The second problem: Broken backtests.**

Your strategy enters at $1,850.23, exits at $1,850.89 (66-pip window). Stop-loss at $1,849.95. The hourly candle shows `[O: 1850.10, H: 1851.20, L: 1849.80, C: 1850.95]`. 

Did you hit take-profit at $1,850.89? Or did price drop to $1,849.95 first and stop you out? **The candle doesn't tell you.** You can't calculate actual risk, can't validate your strategy, can't trust your backtest. A single candle spanning $10 might hide a flash crash that stopped you out—but OHLC makes it look like smooth sailing.

If you're serious about modeling market dynamics, you need tick-level data. Everything else is a lossy approximation.

### Tools That Don't Scale

You decide you need tick data. After searching, you find Dukascopy has high-quality data. You search GitHub for "dukascopy python" and discover... **a graveyard.**

#### The Abandoned

**Half the repos haven't been touched in 3+ years.** Broken dependencies, no type hints, no tests. READMEs that promise everything but crash on Python 3.12. They worked once, for someone, in 2018.

#### The Fragile

**The active ones weren't built for production ML.** You start downloading 2 years of EUR/USD. Six hours in, your connection drops. Start over from scratch—there's no resume logic. Or manually parse logs, reverse-engineer the architecture, write recovery scripts, and *pray* they handle partial state correctly.

#### The Inefficient

**Memory management is an afterthought.** Some libraries load entire datasets into RAM before processing. Fine for a weekend of data. Catastrophic when you need 10 years across 30 symbols.

**Decoding is painfully slow.** Single-threaded, pure-Python loops processing millions of ticks. Every inefficiency compounds at scale. What should take minutes takes hours.

> I duct-taped together scripts for a while—custom resume logic, manual retry handling, homegrown decoders. Then I got tired of maintaining infrastructure when I should have been training models.

### The Bandwidth Bottleneck

Dukascopy's datafeed is free and high-quality. It's also **aggressively rate-limited.**

You can download 5-10 requests/second before hitting `429` (rate limit) or `503` (service unavailable) errors. Fine for a week of one symbol. A nightmare when you need:
```
30 currency pairs
× 20 years of history each  
× 24 hourly chunks per day
─────────────────────────────
= 5,256,000 individual requests
```

At 5 requests/second with perfect uptime: **12 days of continuous downloading.**

In practice, with retries, backoff delays, and connection issues? **Closer to 3 weeks.**

And that's with a *single* connection. Most libraries don't support concurrent downloads, let alone load distribution. You're stuck babysitting a script for weeks, hoping nothing crashes overnight.

> **The bottleneck isn't your internet connection. It's the architecture of the tools.**

### The Cost vs Quality Tradeoff

You could just *pay* for data. Plenty of vendors sell tick data—$X/month, enterprise contracts with minimums.

For a hedge fund? A rounding error. For an independent researcher, grad student, or someone in a country where **$500/month is twice a salary**? A non-starter.

And you're often paying for **convenience, not quality.** Many paid APIs just resell the same Dukascopy data you could get free, wrapped in a nicer interface. You're paying someone else to solve the infrastructure problem.

Which would be fine—if the free tools actually worked. But they don't.

**The False Trilemma:**

1. 💰 **Expensive APIs** — Solve the problem but price out independent research
2. 🔓 **Free data** — High-quality but inaccessible without building infrastructure  
3. 🗑️ **Resampled data** — Easy to get but useless for serious work

> **There should be a fourth option:** Free, high-quality, and actually usable at scale.

> **That's TickVault.**

---

## 🏗️ The Design Philosophy

I built TickVault around three core principles:

### 1) Mirror Dukascopy's Structure 1:1

No reformatting, no "clever" reorganization. The filesystem layout matches the source URLs exactly. No surprises when you need to debug. **Single source of truth.**

### 2) Store Raw, Compressed Data

Keep the original `.bi5` files as-is. Change your resampling strategy in 6 months? No re-downloading terabytes. Need to reproduce results from a year ago? The data hasn't been pre-processed into irrelevance.

**ELT over ETL** — extract and store, transform when you need it.

### 3) Decode On-Demand

Want one day of EUR/USD? Decompress those 24 hourly chunks. Want 5 years of 30 symbols? Same code path, just more chunks. Memory usage stays constant because you're not loading everything upfront.

---

## 🚀 Quick Start

These principles translate into a simple, powerful API. Here's what that looks like in code:
```bash
pip install tick-vault
```

Download a month of gold tick data:
```python
from datetime import datetime
from tick_vault import download_range

await download_range(
    symbol='XAUUSD',
    start=datetime(2024, 1, 1),
    end=datetime(2024, 2, 1)
)
```
> By default the data would be stored in "tick_vault_data" directory in you current working directory. You can change this default behavior. More on this in the **"Configuration"** section bellow, or refer to the full configuration table and details in the [repo](https://github.com/keyhankamyar/TickVault).

Read it back as a pandas DataFrame:
```python
from tick_vault import read_tick_data

df = read_tick_data(
    symbol='XAUUSD',
    start=datetime(2024, 1, 1),
    end=datetime(2024, 2, 1)
)

print(df.head())
# time                     ask        bid      ask_volume  bid_volume
# 2024-01-01 00:00:01.234  2062.450  2062.430  150         230
# ...
```

---

## 💡 How TickVault Works Differently

I didn't want to build "yet another Dukascopy wrapper." I wanted to solve the **underlying architectural problems** that make existing tools fragile.

**The goal:** Minimal, Pythonic, type-safe, performant, and scalable.

### Store Raw, Mirror 1:1

TickVault's filesystem structure mirrors Dukascopy's URL structure **exactly:**
```
tick_vault_data/
└── downloads/
    └── XAUUSD/
        └── 2024/
            └── 02/          # Month: 0-indexed (00=Jan, 11=Dec)
                └── 15/      # Day
                    ├── 00h_ticks.bi5
                    ├── 01h_ticks.bi5
                    └── ...
```

Every file is stored in its original compressed `.bi5` format. No reformatting, no "clever" reorganization, no pre-processing.

**Why this matters:**

✅ **Single source of truth** — When something breaks, you know exactly where to look. The file at `XAUUSD/2024/02/15/14h_ticks.bi5` corresponds to `https://datafeed.dukascopy.com/datafeed/XAUUSD/2024/02/15/14h_ticks.bi5`. No mental mapping required.

✅ **Reproducibility** — Your resampling strategy from 6 months ago produced different results today? The raw data hasn't changed—you can investigate. With pre-processed data, you're just guessing.

✅ **Storage efficiency** — Compressed tick data is surprisingly small. 20 years of Gold (200,000+ hourly files) is **<15GB**. Keep the originals, transform on-demand.

✅ **Future-proof** — Need millisecond timestamps instead of second-level precision? The raw ticks are still there. You're not locked into past decisions.

> **ELT over ETL** — Extract-Load-Transform instead of Extract-Transform-Load. Get the data once, transform it however many times you need.

### Metadata-Driven Resume

Every download attempt gets tracked in a SQLite database:
```
metadata.db
├── symbol_XAUUSD
│   ├── timestamp: 1704067200  (2024-01-01 00:00 UTC)
│   ├── has_data: 1
│   └── ...
└── symbol_EURUSD
    └── ...
```

Each symbol gets its own table. Each hour gets a row with two pieces of information:
1. Did we attempt to download it?
2. Does data exist for this hour? (Some hours legitimately have no data—weekends for forex, market holidays, etc.)

**Why this matters:**

✅ **True resume capability** — Download crashes at hour 50,000? Run the same command again. TickVault checks the database, skips completed work, continues where it left off. Zero manual state tracking.

✅ **Incremental updates** — Downloaded through March? Now it's June? Set `end=datetime.now()` and TickVault only fetches new chunks. Historical data stays untouched.

✅ **Gap detection** — Before reading, TickVault verifies continuity. Missing hours in your range? It tells you *before* you waste time on a broken dataset.

✅ **Producer-consumer pattern** — Download workers (producers) fetch chunks concurrently and push to a queue. A single metadata worker (consumer) batches database writes. This avoids lock contention while maintaining consistency—even with 500 parallel workers.

### Parallel Everything

TickVault's download architecture is built around concurrency:
```
Orchestrator
├── Proxy A → Worker 1, Worker 2, ..., Worker N
├── Proxy B → Worker 1, Worker 2, ..., Worker N
└── Proxy C → Worker 1, Worker 2, ..., Worker N
     ↓
  Result Queue
     ↓
  Metadata Worker (single writer)
```

Each proxy gets its own pool of async workers (default: 10 per proxy). Each worker:
1. Pulls a chunk from the work queue
2. Fetches it via `httpx` with retry logic
3. Saves the compressed data to disk
4. Reports the result to the metadata worker

**Why this matters:**

**Speed** — With 7 proxies × 10 workers = 70 concurrent requests. That 12-day download? **Done in hours.**

**Rate limit mitigation** — Dukascopy rate-limits per IP. Distributing across proxies means you're not constantly hitting limits and backing off.

**Fault tolerance** — One worker hangs? The other 69 keep going. One proxy blocked? Its workers fail gracefully while others continue. The orchestrator handles backpressure—if downloads outpace metadata writes, the queue fills and workers naturally slow down.

**Exponential backoff with context** — Transient network error? Retry with increasing delays. Rate limit with `Retry-After` header? Respect it. Forbidden/blocked? Fail fast, stop wasting time.

### Decode On-Demand

TickVault doesn't pre-process anything. The `.bi5` files stay compressed on disk until you actually need them.

**When you call `read_tick_data()`:**

1. Query the metadata database for available chunks in your time range
2. Verify there are no gaps (fail fast if data is incomplete)
3. Load each compressed chunk sequentially
4. Decompress with LZMA, decode with NumPy structured arrays, fully vectorized

**Why this matters:**

✅ **Flexibility** — Resample to 5-minute bars today, 1-second bars tomorrow? The raw ticks are still there. Calculate VWAP using actual volumes? You have the data. Every transformation is non-destructive.

✅ **Fast enough** — Decompression and decoding are fast—LZMA is optimized, NumPy handles binary parsing efficiently. For most use cases, the bottleneck is your analysis code, not the data loading.

✅ **Coming soon** — Incremental decode-to-database pipelines. Stream chunks directly to SQLite or HDF5 for efficient querying without loading into memory. Same raw source files, different materialization strategies—pick what fits your workflow. Store N-TB compressed, work with 1GB at a time. Same code for one day or ten years—memory usage stays constant.

> **The pattern:** Download once, transform many times. Keep the highest-resolution version, derive everything else as needed.

---

## 👣 Getting Started

### Install
```bash
pip install tick-vault
```

### Basic Workflow

Download and read tick data:
```python
from datetime import datetime
from tick_vault import download_range, read_tick_data

# Download a month of gold tick data
await download_range(
    symbol='XAUUSD',
    start=datetime(2024, 1, 1),
    end=datetime(2024, 2, 1)
)

# Read it back as a pandas DataFrame
df = read_tick_data(
    symbol='XAUUSD',
    start=datetime(2024, 1, 1),
    end=datetime(2024, 2, 1)
)

print(df.head())
# time                     ask        bid      ask_volume  bid_volume
# 2024-01-01 00:00:01.234  2062.450  2062.430  1500000    2300000
```

### Speed It Up with Proxies
```python
# Use multiple proxies to distribute load and avoid rate limits
await download_range(
    symbol='EURUSD',
    start=datetime(2000, 1, 1),
    end=None, # Or datetime.now() for the same effect
    proxies=[
        'http://proxy1.example.com:8080',
        'http://proxy2.example.com:8080',
        'http://proxy3.example.com:8080'
    ]
)
```

### Configuration
```python
from tick_vault import reload_config

# Customize settings
reload_config(
    base_directory='./my_tick_data',
    worker_per_proxy=15,                # More workers per proxy
    fetch_max_retry_attempts=5,         # More retries for flaky connections
    metadata_update_batch_size=200      # Larger batches for efficiency
)
```

Or use environment variables:
```bash
export TICK_VAULT_BASE_DIRECTORY=/data/ticks
export TICK_VAULT_WORKER_PER_PROXY=15
export TICK_VAULT_CONSOLE_LOG_LEVEL=INFO
```

> Full configuration reference in the [repo](https://github.com/keyhankamyar/TickVault)  

### Resuming and Incremental Updates
```python
# Download interrupted? Just run it again
await download_range(
    symbol='XAUUSD',
    start=datetime(2020, 1, 1),
    end=datetime(2024, 1, 1)
)
# TickVault checks metadata, skips completed chunks, resumes where it left off

# Update with recent data
await download_range(
    symbol='XAUUSD',
    start=datetime(2024, 3, 1),
    end=datetime.now()
)
# Only fetches the new chunks, leaves historical data untouched
```

### Data Validation
```python
# Strict mode (default): ensures the exact provided range is present
df = read_tick_data(
    symbol='XAUUSD',
    start=datetime(2024, 1, 1),
    end=datetime(2024, 2, 1),
    strict=True  # Raises error if any hours are missing
)

# Non-strict mode: clips to available data range, still raises error if there are gaps between
df = read_tick_data(
    symbol='XAUUSD',
    start=datetime(2020, 1, 1),  # May be before first available
    end=datetime(2030, 1, 1),    # May be after last available
    strict=False  # Automatically adjusts to what exists
)
```

### Working with Metadata
```python
from tick_vault.metadata import MetadataDB

with MetadataDB() as db:
    # Check what's available
    first = db.first_chunk('XAUUSD')
    last = db.last_chunk('XAUUSD')
    print(f"Data range: {first.time} to {last.time}")
    
    # Find what needs downloading
    pending = db.find_not_attempted_chunks(
        symbol='EURUSD',
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1)
    )
    print(f"{len(pending)} chunks remaining")
    
    # Verify continuity
    db.check_for_gaps(
        symbol='XAUUSD',
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1)
    )  # Raises RuntimeError if gaps exist
```

### Custom Price Scales
```python
# For symbols not in the built-in registry
df = read_tick_data(
    symbol='CUSTOM_PAIR',
    start=datetime(2024, 1, 1),
    end=datetime(2024, 2, 1),
    pipet_scale=0.01  # Specify your own scaling factor
)
```

---

## 🧭 What's Next

TickVault works well for my needs, but there's room to grow. Here's the roadmap:

### Download Pipeline
- Dynamic worker auto-balancing with throughput monitoring
- Adaptive scaling: gradually increase workers until throughput plateaus, then back off
- Async stop events for cleaner worker termination (currently uses sentinel values)

### Reading Pipeline
- Multi-threading and multi-processing support for parallel decoding (helpful for rapid SSDs)
- Streaming decode-to-SQLite pipeline for memory-efficient querying
- HDF5 storage backend option for large datasets

### Developer Experience
- CLI interface for common operations (`tickvault download XAUUSD --start 2024-01-01`)
- Comprehensive pytest test suite

### General Improvements
- Reorganized module structure as the codebase grows
- Unified `download_and_read()` convenience function
- More symbols added to the pipet scale registry

> **The core is stable.** These are refinements, not fundamental changes. If you have ideas or want to contribute, issues and PRs are welcome.

---

## ▶️ Try It Out

TickVault is [open source on GitHub](https://github.com/keyhankamyar/TickVault). If you've fought with financial data pipelines, give it a shot:
```bash
pip install tick-vault
```

---

**Star the repo** if it solves a problem for you — Stars help other researchers discover tools that actually work.

**Open an issue or PR** if you find bugs or have ideas — The codebase is minimal and clean enough to understand in an afternoon.

**Follow along** for more posts on the RL research this was built for, performance optimizations, and lessons from building production data pipelines.

---

## 💭 Final Thoughts

I built TickVault because I needed tick data for reinforcement learning research and nothing else worked well enough.

If you're in the same boat—tired of resampled data, broken tools, and expensive APIs—**this might help.**

---

[![GitHub](https://img.shields.io/badge/GitHub-keyhankamyar%2FTickVault-181717?style=for-the-badge&logo=github)](https://github.com/keyhankamyar/TickVault)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-keyhan--kamyar-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/keyhan-kamyar/)