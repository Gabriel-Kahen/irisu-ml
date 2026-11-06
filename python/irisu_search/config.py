"""Frozen campaign settings; budgets are wall time, never inferred speedups."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CampaignConfig:
    engine: tuple[str, ...]
    oracle: str
    oracle_library: str = ""
    cuda_device: str = "0"
    initial_seeds: int = 128
    seed_generator: int = 0
    attempts_per_seed: int = 3
    control_seed: int = 1298144938
    bootstrap_ticks: int = 2048
    horizon: int = 384
    candidates: int = 16
    survivors: int = 3
    job_seconds: float = 30.0
    engine_timeout: float = 120.0
    maximum_ticks: int = 160000
    checkpoint_limit: int = 1024
    archive_bytes: int = 268435456
    per_cell: int = 3
    promoted_seeds: int = 16
    finalists: int = 4
    minimum_seed_seconds: float = 30.0
    finalist_seconds: float = 120.0
    exploration_fraction: float = .25
    model: str = ""
    train_every_jobs: int = 0
    training_epochs: int = 40
    max_training_examples: int = 4000
    discover_new_seeds: bool = True
    combo_search: bool = False
    combo_frontier_width: int = 4
    combo_terminal_horizon: int = 32768
    combo_root_limit: int = 64
    combo_root_lead: int = 48
    combo_min_chain: int = 3
    seed_cohort_size: int = 8
    lanes: dict[str, float] = field(default_factory=lambda: {
        "seed": .30, "combo": .45, "probe": .15, "wildcard": .10,
    })

    def __post_init__(self):
        if not isinstance(self.engine, (tuple, list)) or not self.engine or any(not isinstance(x, str) or not x for x in self.engine):
            raise ValueError("engine must be a nonempty argument vector")
        object.__setattr__(self, "engine", tuple(self.engine))
        if self.engine and "/" in self.engine[0]:
            object.__setattr__(self, "engine", (str(Path(self.engine[0]).expanduser().resolve()), *self.engine[1:]))
        for name in ("oracle", "oracle_library", "model"):
            if getattr(self, name):
                object.__setattr__(self, name, str(Path(getattr(self, name)).expanduser().resolve()))
        if not self.engine or any(not isinstance(x, str) or not x for x in self.engine):
            raise ValueError("engine must be a nonempty argument vector")
        if type(self.train_every_jobs) is not int or self.train_every_jobs < 0:
            raise ValueError("train_every_jobs must be a nonnegative integer")
        if type(self.combo_search) is not bool:
            raise ValueError("combo_search must be boolean")
        if type(self.discover_new_seeds) is not bool:
            raise ValueError("discover_new_seeds must be boolean")
        if not isinstance(self.cuda_device, str):
            raise TypeError("cuda_device must be a CUDA device selector string")
        if not self.oracle:
            raise ValueError("an independent native oracle is required")
        for name in ("initial_seeds", "attempts_per_seed", "bootstrap_ticks", "horizon",
                     "candidates", "survivors", "maximum_ticks", "checkpoint_limit",
                     "archive_bytes", "per_cell", "promoted_seeds", "finalists", "training_epochs", "max_training_examples",
                     "combo_frontier_width", "combo_terminal_horizon", "combo_root_limit", "combo_root_lead",
                     "combo_min_chain", "seed_cohort_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_seeds > 0xffffffff:
            raise ValueError("initial_seeds exceeds the available uint32 seed domain")
        if self.combo_terminal_horizon > 100000:
            raise ValueError("combo_terminal_horizon must be <= 100000")
        if self.combo_search and self.combo_frontier_width > self.survivors:
            raise ValueError("combo_frontier_width must not exceed survivors")
        if self.horizon > 100000 or self.bootstrap_ticks > 100000:
            raise ValueError("horizon and bootstrap_ticks must be <= 100000")
        if self.training_epochs > 100000:
            raise ValueError("training_epochs must be <= 100000")
        if self.survivors > self.candidates or self.candidates > 1024:
            raise ValueError("require survivors <= candidates <= 1024")
        if type(self.control_seed) is not int or not 0 <= self.control_seed <= 0xffffffff:
            raise ValueError("control_seed must fit uint32")
        if type(self.seed_generator) is not int:
            raise ValueError("seed_generator must be an integer")
        for name in ("job_seconds", "engine_timeout", "minimum_seed_seconds", "finalist_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.exploration_fraction <= 1:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if set(self.lanes) != {"seed", "combo", "probe", "wildcard"} or any(
            not math.isfinite(x) or x <= 0 for x in self.lanes.values()
        ) or abs(sum(self.lanes.values()) - 1) > 1e-9:
            raise ValueError("positive lane shares must sum to one")

    def manifest(self):
        value = asdict(self)
        value["engine"] = list(self.engine)
        return value

    @classmethod
    def parse(cls, value):
        return cls(**value)


def default_paths():
    root = Path(__file__).resolve().parents[2]
    return {
        "engine": str(root / "build-project-exact/irisu-search-engine"),
        "oracle": str(root / "build-project-exact/irisu-game-oracle"),
        "oracle_library": str(root / ".local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so"),
        "incumbent": str(root / "irisu-high-score-4393101.rpy"),
    }
