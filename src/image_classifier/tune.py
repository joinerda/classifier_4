from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Dict, Iterable, List

from .config import TrainConfig
from .train import train_model
from .utils import resolve_seed, set_seed


def tune_hyperparams(
    base_config: TrainConfig,
    grid: Dict[str, Iterable],
    max_trials: int | None = None,
    deterministic: bool | None = None,
) -> List[Dict[str, float]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    trials = list(itertools.product(*values))

    if max_trials is not None:
        trials = trials[:max_trials]

    results: List[Dict[str, float]] = []
    for idx, combo in enumerate(trials, start=1):
        override = dict(zip(keys, combo))
        cfg = replace(base_config, **override)
        if deterministic is not None:
            cfg.deterministic = deterministic

        seed = resolve_seed(cfg.seed)
        set_seed(seed, cfg.deterministic)
        cfg.seed = seed

        metrics = train_model(cfg)
        results.append({"trial": idx, **override, **metrics})

    results.sort(key=lambda r: r.get("best_val_acc", 0.0), reverse=True)
    return results
