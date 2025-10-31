---
title: "Declarative, Conditional Search Spaces - The Missing Tool for HPO"
date: 2025-01-20T12:00:00+03:30
draft: false
tags: ["machine-learning", "hyperparameter-optimization", "neural-architecture-search", "python", "research-tools", "open-source"]
categories: ["Projects"]
description: "Announcing SpaX, a Python library that helps you define type-checked, conditional search spaces, visualize and refine them over time—without rewriting your code between experiments."
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
summary: "Type-safe, declarative search spaces for real-world HPO. Define once, refine safely. Conditional, nested spaces with minimal syntax and clear semantics."
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

---

### A more complete example: before vs after

> **Heads-up:** The next block is intentionally verbose to mirror real-world wiring. If you want the clean version, skip to [“After: SpaX.”](#after-spax)

#### Before: Pydantic + Optuna (manual wiring, fragile)

```python
# Baseline: pure Pydantic config + manual Optuna suggestions
# Nested + conditional + polymorphic (Dense vs Conv) handled by hand.

from typing import Literal
from pydantic import BaseModel, Field
import optuna


# --- Configs (validation only; no search-space semantics) --------------------

class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd"]
    learning_rate: float = Field(ge=1e-5, le=1e-2)
    # conditionals (must be enforced in code)
    momentum: float | None = None   # only for SGD
    beta2: float | None = None      # only for Adam


class DenseConfig(BaseModel):
    num_layers: int = Field(ge=2, le=12)
    activation: Literal["silu", "mish", "relu", "gelu"]
    norm_input: bool
    use_dropout: bool
    dropout_rate: float | None = None  # only if use_dropout


class ConvConfig(BaseModel):
    num_layers: int = Field(ge=2, le=12)
    activation: Literal["silu", "mish", "relu", "gelu"]
    kernel_size: int = Field(ge=1, le=64)
    norm_input: bool
    use_dropout: bool
    dropout_rate: float | None = None  # only if use_dropout


class ModelConfig(BaseModel):
    # Polymorphic choice: DenseConfig OR ConvConfig (Pydantic won't choose for you)
    block_type: Literal["dense", "conv"]
    dense: DenseConfig | None = None
    conv: ConvConfig | None = None
    num_blocks: int = Field(ge=2, le=12)
    # rule lives outside the model; you must remember to enforce it later
    # use_checkpoint: bool


class TrainingConfig(BaseModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    batch_size: int = Field(ge=16, le=128)


# --- Objective with hand-rolled suggest_* logic and wiring -------------------

def objective(trial: optuna.Trial) -> float:
    # Optimizer
    opt_name = trial.suggest_categorical("optimizer.name", ["adam", "sgd"])
    lr = trial.suggest_float("optimizer.learning_rate", 1e-5, 1e-2, log=True)

    if opt_name == "sgd":
        momentum = trial.suggest_float("optimizer.momentum", 0.0, 0.99)
        beta2 = None
    else:
        beta2 = trial.suggest_float("optimizer.beta2", 0.9, 0.999)
        momentum = None

    # Polymorphic block choice
    block_type = trial.suggest_categorical("model.block_type", ["dense", "conv"])

    if block_type == "dense":
        num_layers = trial.suggest_int("model.dense.num_layers", 2, 12)
        activation = trial.suggest_categorical(
            "model.dense.activation", ["silu", "mish", "relu", "gelu"]
        )
        norm_input = trial.suggest_categorical("model.dense.norm_input", [True, False])
        use_dropout = trial.suggest_categorical("model.dense.use_dropout", [True, False])
        dropout_rate = (
            trial.suggest_float("model.dense.dropout_rate", 0.05, 0.5)
            if use_dropout else None
        )
        dense = DenseConfig(
            num_layers=num_layers,
            activation=activation,
            norm_input=norm_input,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        conv = None
    else:
        num_layers = trial.suggest_int("model.conv.num_layers", 2, 12)
        activation = trial.suggest_categorical(
            "model.conv.activation", ["silu", "mish", "relu", "gelu"]
        )
        kernel_size = trial.suggest_int("model.conv.kernel_size", 1, 64)
        norm_input = trial.suggest_categorical("model.conv.norm_input", [True, False])
        use_dropout = trial.suggest_categorical("model.conv.use_dropout", [True, False])
        dropout_rate = (
            trial.suggest_float("model.conv.dropout_rate", 0.05, 0.5)
            if use_dropout else None
        )
        conv = ConvConfig(
            num_layers=num_layers,
            activation=activation,
            kernel_size=kernel_size,
            norm_input=norm_input,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        dense = None

    num_blocks = trial.suggest_int("model.num_blocks", 2, 12)
    batch_size = trial.suggest_int("batch_size", 16, 128)

    config = TrainingConfig(
        model=ModelConfig(
            block_type=block_type, dense=dense, conv=conv, num_blocks=num_blocks
        ),
        optimizer=OptimizerConfig(
            name=opt_name, learning_rate=lr, momentum=momentum, beta2=beta2
        ),
        batch_size=batch_size,
    )

    # Hidden rule: enable checkpointing for deep models (easy to forget/duplicate)
    use_checkpoint = num_blocks > 8

    return train_and_evaluate(config, use_checkpoint=use_checkpoint)
```

**Where this bites you:**

* **Naming fragility:** `model.dense.dropout_rate` vs `model.conv.dropout_rate`. Typos silently create *new* parameters; later analysis breaks.
* **Conditional drift:** You must reset inactive fields (`momentum=None` when not SGD, `dropout_rate=None` when no dropout). Easy to forget in either direction.
* **Dead branches & wasted compute:** If you forget the `if use_dropout:` guard or misname it, Optuna happily “tunes” a parameter that doesn’t matter.
* **Rules hidden in code:** `use_checkpoint = num_blocks > 8` lives outside the model; it’s not validated or visible in a canonical space view.
* **No canonical space description:** There’s no ground-truth tree of “what is tunable under which conditions?”
* **Polymorphism boilerplate:** You manually branch Dense vs Conv, duplicate naming, and keep both schemas in sync.
* **Polymorphic deserialization ambiguity (Pydantic):** Union-like fields (`DenseCfg | ConvCfg`) don’t round-trip cleanly. When you load a saved config or trial params, Pydantic can’t infer **which** variant a parameter belongs to unless you hand-roll a discriminator (`block_type`) and ensure only one sub-config is non-null. It’s easy to mis-reconstruct past runs.
* **Cross-branch name collisions & accidental coupling:** It’s tempting to reuse the same key for different branches (e.g., `dropout_rate` or `num_layers`). But they are **not the same space**—a Dense block might prefer deeper nets than a Conv block (or vice versa), and ranges/semantics differ. Reusing names (or forgetting the branch prefix) couples unrelated spaces, corrupts analysis, and can lead the optimizer to mix signals across architectures.

---

<a id="after-spax"></a>
#### After: SpaX (declarative, conditional, polymorphic, canonical)

```python
from typing import Literal

import optuna
from pydantic import Field

import spax as sp


class OptimizerConfig(sp.Config):
    name: Literal["adam", "sgd"]
    learning_rate: float = sp.Float(ge=1e-5, le=1e-2, distribution="log")

    momentum: float | None = sp.Conditional(
        sp.FieldCondition("name", sp.EqualsTo("sgd")),
        true=sp.Float(ge=0.0, le=0.99),
        false=None,  # only for SGD
    )

    beta2: float | None = sp.Conditional(
        sp.FieldCondition("name", sp.EqualsTo("adam")),
        true=sp.Float(ge=0.9, le=0.999),
        false=None,  # only for Adam
    )


class DenseConfig(sp.Config):
    num_layers: int = Field(ge=2, le=12)
    activation: Literal["silu", "mish", "relu", "gelu"]
    norm_input: bool
    use_dropout: bool

    dropout_rate: float | None = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(ge=0.05, le=0.5),  # only if use_dropout
        false=None,
    )


class ConvConfig(sp.Config):
    num_layers: int = Field(ge=2, le=12)
    activation: Literal["silu", "mish", "relu", "gelu"]
    kernel_size: int = Field(ge=1, le=64)
    norm_input: bool
    use_dropout: bool

    dropout_rate: float | None = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(ge=0.05, le=0.5),  # only if use_dropout
        false=None,
    )


class ModelConfig(sp.Config):
    # Polymorphic field: either DenseConfig or ConvConfig.
    # This automatically becomes sp.Categorical([DenseConfig, ConvConfig])
    # SpaX handles type tagging automatically for (de)serialization.
    block_config: DenseConfig | ConvConfig
    num_blocks: int = Field(ge=2, le=12)

    # Rule is part of the config, not hidden in training code
    use_checkpoint: bool = sp.Conditional(
        sp.FieldCondition("num_blocks", sp.LargerThan(8)),
        true=True,
        false=False,
    )


class TrainingConfig(sp.Config):
    model: ModelConfig
    optimizer: OptimizerConfig
    batch_size: int = Field(ge=16, le=128)


def objective(trial: optuna.Trial) -> float:
    # One line: validated, conditional, nested sampling
    return train_and_evaluate(TrainingConfig.from_trial(trial))
```

##### Canonical view of the space:
```python
print(TrainingConfig.get_tree())
```
<details><summary>Click to expand/collapse</summary>

```text
TrainingConfig
├─ model: ModelConfig
│  ├─ block_config: Categorical
│  │  ├─ DenseConfig
│  │  │  ├─ num_layers: Int([2, 12], uniform)
│  │  │  ├─ activation: Categorical
│  │  │  │  ├─ 'silu'
│  │  │  │  ├─ 'mish'
│  │  │  │  ├─ 'relu'
│  │  │  │  └─ 'gelu'
│  │  │  ├─ norm_input: Categorical
│  │  │  │  ├─ True
│  │  │  │  └─ False
│  │  │  ├─ use_dropout: Categorical
│  │  │  │  ├─ True
│  │  │  │  └─ False
│  │  │  └─ dropout_rate: Conditional (if use_dropout == True)
│  │  │     ├─ true: Float([0.05, 0.5], uniform)
│  │  │     └─ false: None
│  │  └─ ConvConfig
│  │     ├─ num_layers: Int([2, 12], uniform)
│  │     ├─ activation: Categorical
│  │     │  ├─ 'silu'
│  │     │  ├─ 'mish'
│  │     │  ├─ 'relu'
│  │     │  └─ 'gelu'
│  │     ├─ kernel_size: Int([1, 64], uniform)
│  │     ├─ norm_input: Categorical
│  │     │  ├─ True
│  │     │  └─ False
│  │     ├─ use_dropout: Categorical
│  │     │  ├─ True
│  │     │  └─ False
│  │     └─ dropout_rate: Conditional (if use_dropout == True)
│  │        ├─ true: Float([0.05, 0.5], uniform)
│  │        └─ false: None
│  ├─ num_blocks: Int([2, 12], uniform)
│  └─ use_checkpoint: Conditional (if num_blocks > 8)
│     ├─ true: True
│     └─ false: False
├─ optimizer: OptimizerConfig
│  ├─ name: Categorical
│  │  ├─ 'adam'
│  │  └─ 'sgd'
│  ├─ learning_rate: Float([1e-05, 0.01], log)
│  ├─ momentum: Conditional (if name == 'sgd')
│  │  ├─ true: Float([0.0, 0.99], uniform)
│  │  └─ false: None
│  └─ beta2: Conditional (if name == 'adam')
│     ├─ true: Float([0.9, 0.999], uniform)
│     └─ false: None
└─ batch_size: Int([16, 128], uniform)
```
</details>

##### Hierarchical parameter names:

```python
print(TrainingConfig.get_parameter_names())
```
<details><summary>Click to expand/collapse</summary>

```text
[
    "TrainingConfig.model::ModelConfig.block_config",
    "TrainingConfig.model::ModelConfig.block_config::DenseConfig.activation",
    "TrainingConfig.model::ModelConfig.block_config::DenseConfig.norm_input",
    "TrainingConfig.model::ModelConfig.block_config::DenseConfig.num_layers",
    "TrainingConfig.model::ModelConfig.block_config::DenseConfig.use_dropout",
    "TrainingConfig.model::ModelConfig.block_config::DenseConfig.dropout_rate::true_branch",
    "TrainingConfig.model::ModelConfig.block_config::ConvConfig.activation",
    "TrainingConfig.model::ModelConfig.block_config::ConvConfig.kernel_size",
    "TrainingConfig.model::ModelConfig.block_config::ConvConfig.norm_input",
    "TrainingConfig.model::ModelConfig.block_config::ConvConfig.num_layers",
    "TrainingConfig.model::ModelConfig.block_config::ConvConfig.use_dropout",
    "TrainingConfig.model::ModelConfig.block_config::ConvConfig.dropout_rate::true_branch",
    "TrainingConfig.model::ModelConfig.num_blocks",
    "TrainingConfig.optimizer::OptimizerConfig.learning_rate",
    "TrainingConfig.optimizer::OptimizerConfig.name",
    "TrainingConfig.optimizer::OptimizerConfig.beta2::true_branch",
    "TrainingConfig.optimizer::OptimizerConfig.momentum::true_branch",
    "TrainingConfig.batch_size",
]
```
</details>

<div aria-hidden="true" style="height: 1.25rem;"></div>

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