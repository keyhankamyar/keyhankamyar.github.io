---
title: "300 Parameters and No Plan? Meet SpaX"
date: 2025-01-20T12:00:00+03:30
draft: false
tags: ["machine-learning", "hyperparameter-optimization", "neural-architecture-search", "python", "research-tools", "open-source"]
categories: ["Projects"]
description: "Here's what nobody tells you about hyperparameter optimization and neural architecture search at scale: you don't search 300+ parameters at once. SpaX is built around iterative refinement - the workflow that actually works. Type-safe search spaces, conditional parameters, progressive narrowing with overrides, zero boilerplate, seamless Optuna integration. The Python HPO/NAS library that should have existed from day one, for ML researchers exploring complex hyperparameter spaces."
images:
  - /images/spax-og.png
keywords: 
  - hyperparameter optimization
  - neural architecture search
  - HPO python library
  - NAS configuration
  - search space definition
  - conditional parameters
  - iterative hyperparameter tuning
  - optuna integration
  - pydantic configuration
  - machine learning experiments
  - type-safe ML configs
  - automl tools
  - hyperparameter search spaces
  - ML experiment tracking
  - research configuration tools
  - python ML library
  - hyperparameter tuning framework
  - deep learning optimization
  - reproducible ML research
  - open source ML tools
author: "Keyhan Kamyar"
canonicalURL: "https://keyhankamyar.github.io/posts/spax-introduction/"
summary: "Type-safe search space definition for ML. One-line Pydantic migration, conditional parameters, iterative refinement with overrides, seamless Optuna integration. The HPO/NAS library that should have existed from day one."
---

Finding optimal parameters—hyperparameters, architecture choices, whatever—should be straightforward. It's not. Which libraries? What's the actual workflow? How do I search a massive space without wasting weeks of compute? By the time you figure it out, you've written hundreds of lines of boilerplate, debugged silent parameter conflicts, and discovered that "search everything at once" wastes compute and gets bad results.

> **TL;DR:** SpaX is a Python library for type-safe search space definition and exploration. One-line migration from Pydantic gets you automatic space inference, conditional parameters, random sampling, and Optuna integration. Progressive refinement via override files—narrow your search space between experiments without touching code. Designed to integrate with your stack, not replace it.

**If you've been doing this for years:** You already know the workflow—iterative refinement. Don't search 300 parameters at once; fix some to reasonable defaults, explore a subset, narrow promising regions, then progressively expand. But current tools make this painful: editing code, changing ranges across files, tracking explorations in spreadsheets. SpaX makes this workflow first-class with override files, type-safe validation, and space visualization—all without touching your code between iterations.

**If you're new to HPO/NAS:** Here's what tutorials won't tell you—you don't plug Optuna into a 200-parameter space and hope. The real workflow is iterative refinement: fix most parameters, explore a few, narrow what works, then expand. SpaX gives you this workflow built-in, with clear patterns for defining spaces, catching invalid configurations early, and refining without rewriting code.

---

## Enter SpaX

[![GitHub](https://img.shields.io/badge/GitHub-SpaX-181717?style=flat-square&logo=github)](https://github.com/keyhankamyar/SpaX)
[![PyPI](https://img.shields.io/pypi/v/spax?style=flat-square)](https://pypi.org/project/spax/)
```bash
pip install spax
```

SpaX is a Python library for declarative search space definition and exploration. Start simple - if you're already using Pydantic, it's often just changing `BaseModel` to `Config` and you get automatic space inference, space visualization, random sampling for quick sanity checks, automatic serialization of complex spaces, and Optuna integration. Need more? Add conditional parameters, nested configs, polymorphic fields, and iterative refinement with the same clean API. Define your search space once, explore it intelligently.

Here's what the simple case looks like. A typical Pydantic config for an experiment:
```python
from pydantic import BaseModel, Field
from typing import Literal

class ExperimentConfig(BaseModel):
    learning_rate: float = Field(ge=1e-5, le=1e-1)
    batch_size: int = Field(ge=16, le=128)
    optimizer: Literal["adam", "sgd", "rmsprop"]
    num_layers: int = Field(ge=1, le=10)
```

Change one line:
```python
import spax as sp

class ExperimentConfig(sp.Config):  # BaseModel → Config
    learning_rate: float = Field(ge=1e-5, le=1e-1)
    batch_size: int = Field(ge=16, le=128)
    optimizer: Literal["adam", "sgd", "rmsprop"]
    num_layers: int = Field(ge=1, le=10)
```

**What you just unlocked:**
```python
# Visualize your search space
print(ExperimentConfig.get_tree())
# ExperimentConfig
# ├─ learning_rate: Float([1e-05, 0.1], log)
# ├─ batch_size: Int([16, 128], uniform)
# ├─ optimizer: Categorical['adam', 'sgd', 'rmsprop']
# └─ num_layers: Int([1, 10], uniform)

# Random sampling for sanity checks
config = ExperimentConfig.random(seed=42)
# ExperimentConfig(learning_rate=2.788e-05, batch_size=97, optimizer='rmsprop', num_layers=7)

# Direct Optuna integration - no more manual suggest_* calls
def objective(trial):
    config = ExperimentConfig.from_trial(trial)  # One line instead of:
    # lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # batch = trial.suggest_int('batch', 16, 128)
    # opt = trial.suggest_categorical('opt', ['adam', 'sgd', 'rmsprop'])
    # layers = trial.suggest_int('layers', 1, 10)
    return train_and_evaluate(config)

study.optimize(objective, n_trials=100)
```

> 💡 **No DSL to learn.** If you know Pydantic and type hints, you already know SpaX. The spaces are just Python classes with clear semantics.

This is just automatic inference from type hints and `Field()` constraints. But what about parameters that *depend on other parameters*? What about configs that need to encode your domain knowledge and best practices? That's where SpaX gets interesting.

---

**Sometimes automatic inference isn't enough - use explicit spaces for full control:**
```python
class TrainingConfig(sp.Config):
    # Log distribution for learning rates (better for HPO)
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    
    # Weighted categorical - bias toward Adam without hard-fixing it
    optimizer: str = sp.Categorical([
        sp.Choice("adam", weight=3.0),  # 3x more likely
        sp.Choice("sgd", weight=1.0),
        sp.Choice("rmsprop", weight=1.0),
    ])
```

When you need log distributions, weighted sampling, or just want the search space to be immediately obvious in your code—use explicit SpaX spaces.

**But here's where it gets interesting.** What if your parameters should *adapt* based on other choices?
```python
class SimpleConditionalConfig(sp.Config):
    use_dropout: bool
    num_layers: int = sp.Int(ge=1, le=12)
    
    # dropout_rate only exists when use_dropout=True
    dropout_rate: float = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(ge=0.1, le=0.5),
        false=0.0,
    )
    
    # Deep networks need gradient checkpointing
    use_grad_checkpointing: bool = sp.Conditional(
        sp.FieldCondition("num_layers", sp.LargerThan(8)),
        true=True,
        false=False,
    )
```

Your configs now enforce dependencies: you can't accidentally run a 12-layer model without checkpointing, or try to tune dropout when it's disabled. Invalid combinations simply don't exist.

**Here's what you can build with this:**
```python
class SmartTrainingConfig(sp.Config):
    num_encoder_layers: int = sp.Int(ge=4, le=32)
    batch_size: int = sp.Int(ge=8, le=128)
    
    optimizer: str = sp.Categorical([
        sp.Choice("adam", weight=3.0),
        sp.Choice("sgd", weight=1.0),
        sp.Choice("rmsprop", weight=1.0),
    ])
    
    # Deep networks → gradient checkpointing
    use_grad_checkpointing: bool = sp.Conditional(
        sp.FieldCondition("num_encoder_layers", sp.LargerThan(8)),
        true=True,
        false=False,
    )
    
    # Small batches → gradient accumulation
    accumulation_steps: int = sp.Conditional(
        sp.FieldCondition("batch_size", sp.SmallerThan(32, or_equals=True)),
        true=sp.Int(ge=2, le=8),
        false=1,
    )
    
    # Large encoders → smaller heads to balance compute
    num_head_layers: int = sp.Conditional(
        sp.FieldCondition("num_encoder_layers", sp.LargerThan(16)),
        true=sp.Int(ge=1, le=3),
        false=sp.Int(ge=2, le=8),
    )

print(SmartTrainingConfig.get_tree())
# SmartTrainingConfig
# ├─ num_encoder_layers: Int([4, 32], uniform)
# ├─ batch_size: Int([8, 128], uniform)
# ├─ optimizer: Categorical
# │  ├─ 'adam' (weight: 3.0)
# │  ├─ 'sgd'
# │  └─ 'rmsprop'
# ├─ use_grad_checkpointing: Conditional (if num_encoder_layers > 8)
# │  ├─ true: True
# │  └─ false: False
# ├─ accumulation_steps: Conditional (if batch_size <= 32)
# │  ├─ true: Int([2, 8], uniform)
# │  └─ false: 1
# └─ num_head_layers: Conditional (if num_encoder_layers > 16)
#    ├─ true: Int([1, 3], uniform)
#    └─ false: Int([2, 8], uniform)
```

> 🔒 **Invalid configurations don't exist.** SpaX validates dependencies at definition time. You can't build a config that violates your constraints—and neither can your HPO library.

No manual validation. No invalid combinations slipping through. No forgetting to enable checkpointing for deep models. And HPO libraries like Optuna only explore valid configurations—conditional dependencies are handled automatically.

---

Remember iterative refinement? Here's how it works in practice:
```python
# Start with broad search
config = ExperimentConfig.random(seed=42)

# After initial experiments, narrow and focus
override = {
    "learning_rate": [1e-4, 1e-3, 1e-2],  # Grid: sample from specific values
    "batch_size": {"ge": 32, "le": 64},   # Range: narrow bounds
    "optimizer": "adam",                  # Fixed: lock to best
    # num_layers not specified → keeps exploring full range [1, 10]
}

config = ExperimentConfig.random(seed=43, override=override)

# Works with Optuna too
config = ExperimentConfig.from_trial(trial, override=override)

# Visualize your refined space
print(ExperimentConfig.get_tree(override=override))
```

> 📦 **Works with your stack.** SpaX configs serialize to JSON/YAML/TOML. Use them with Hydra, OmegaConf, MLflow—whatever you already have.

No code changes. No redefining spaces. Just progressively refine with override dicts or config files. The base config defines absolute bounds—overrides let you explore subsets without touching your source.

---

## What This Actually Means for Research

This isn't just about cleaner code (though you get that). It's about where your time and mental energy go.

**Time back:** No more writing validation logic for the fifth time. No more debugging why your "best" config from trial 847 is missing from your logs. No more manually editing parameter bounds across multiple files between experiment iterations. The boilerplate that used to take hours per project—gone.

**Better science:** Iterative refinement means you explore your space intelligently instead of hoping 1000 random trials find something. After 200 trials, you see `learning_rate` between 1e-4 and 1e-3 works best—drop an override, narrow that range, explore other parameters. Your compute budget goes further because you're searching strategically. Type-safe configs mean fewer "wait, which parameter values did I use for that run?" moments. Reproducibility by default.

**Mental energy:** Your brain is for research questions, not "did I remember to validate that `batch_size` and `num_gpus` produce a valid total batch size?" or "which file defined that nested optimizer config again?" The cognitive overhead of managing configuration state across experiments just... disappears.

And this is just the foundation. Rich visualizations for search space exploration, automatic pruning of unpromising regions, more HPO framework integrations, experiment tracking—all coming. Build on SpaX now, get the future features as they land.

Define your space once. Explore it properly. Focus on the actual research.

---

## Try It
```bash
pip install spax
```

SpaX is open source and ready to use. The [GitHub repo](https://github.com/keyhankamyar/SpaX) has comprehensive examples—from quick starts to complex nested configs with conditionals. Start with the notebooks in `/examples`—they're designed to get you productive fast.

**Found it useful?** Your feedback matters. Open an issue if something's unclear, breaks, or could be better. Pull requests welcome if you want to contribute.

**Working well for your research?** Share it with your lab. Tweet about it. The more researchers who stop fighting config boilerplate, the better.

Let's spend our time on actual research.

---

[![GitHub](https://img.shields.io/badge/GitHub-SpaX-181717?style=for-the-badge&logo=github)](https://github.com/keyhankamyar/SpaX)
[![PyPI](https://img.shields.io/pypi/v/spax?style=for-the-badge&logo=pypi)](https://pypi.org/project/spax/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)