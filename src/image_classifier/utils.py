import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    return seed


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def run_id() -> Optional[str]:
    return os.getenv("RUN_ID") or os.getenv("SLURM_JOB_ID")


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
