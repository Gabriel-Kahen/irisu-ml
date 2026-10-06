"""Development-only reserve-band ranking for measured planner branches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


VERSION = "irisu-exact-development-reserve-band-comparator-v1"


def _plain_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ReserveBandConfig:
    horizon_ticks: int
    gauge_max: int
    contingency_gauge: int

    def __post_init__(self) -> None:
        _plain_int(self.horizon_ticks, "horizon_ticks", minimum=1)
        _plain_int(self.gauge_max, "gauge_max", minimum=1)
        _plain_int(self.contingency_gauge, "contingency_gauge")

    @property
    def useful_gauge_ceiling(self) -> int:
        return self.gauge_max // 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReserveBandConfig":
        return cls(
            _plain_int(value.get("horizon_ticks"), "horizon_ticks", minimum=1),
            _plain_int(value.get("gauge_max"), "gauge_max", minimum=1),
            _plain_int(value.get("contingency_gauge"), "contingency_gauge"),
        )

    def manifest(self) -> dict[str, int]:
        return {**asdict(self), "useful_gauge_ceiling": self.useful_gauge_ceiling}


@dataclass(frozen=True, slots=True)
class ProbeCandidate:
    candidate_index: int
    survival_ticks: int
    terminated: bool
    truncated: bool
    minimum_gauge: int
    final_gauge: int
    imminent_visible_rot_liability: int
    renewal_clears: int
    score: int
    invalid_actions: int = 0

    def __post_init__(self) -> None:
        for name in (
            "candidate_index",
            "survival_ticks",
            "imminent_visible_rot_liability",
            "renewal_clears",
            "invalid_actions",
        ):
            _plain_int(getattr(self, name), name)
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminated and truncated must be booleans")
        for name in ("minimum_gauge", "final_gauge", "score"):
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), int
            ):
                raise ValueError(f"{name} must be an integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProbeCandidate":
        try:
            return cls(
                candidate_index=value["candidate_index"],
                survival_ticks=value["survival_ticks"],
                terminated=value["terminated"],
                truncated=value["truncated"],
                minimum_gauge=value["minimum_gauge"],
                final_gauge=value["final_gauge"],
                imminent_visible_rot_liability=value[
                    "imminent_visible_rot_liability"
                ],
                renewal_clears=value["renewal_clears"],
                score=value["score"],
                invalid_actions=value.get("invalid_actions", 0),
            )
        except KeyError as exc:
            raise ValueError(f"candidate is missing {exc.args[0]}") from exc

    @property
    def failed(self) -> bool:
        return self.terminated or self.truncated

    def manifest(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: ProbeCandidate
    full_survival: bool
    net_minimum_gauge: int
    net_final_gauge: int
    reserve_solvent: bool
    rank: tuple[int, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.manifest(),
            "full_survival": self.full_survival,
            "net_minimum_gauge": self.net_minimum_gauge,
            "net_final_gauge": self.net_final_gauge,
            "reserve_solvent": self.reserve_solvent,
            "rank": list(self.rank),
        }


def rank_candidate(
    candidate: ProbeCandidate, config: ReserveBandConfig
) -> RankedCandidate:
    if candidate.survival_ticks > config.horizon_ticks:
        raise ValueError("candidate survival exceeds the configured horizon")
    liability = candidate.imminent_visible_rot_liability
    net_minimum = min(candidate.minimum_gauge, config.useful_gauge_ceiling) - liability
    net_final = min(candidate.final_gauge, config.useful_gauge_ceiling) - liability
    full_survival = (
        candidate.survival_ticks == config.horizon_ticks and not candidate.failed
    )
    solvent = (
        full_survival
        and net_minimum >= config.contingency_gauge
        and net_final >= config.contingency_gauge
    )
    rank = (
        int(candidate.invalid_actions == 0),
        candidate.survival_ticks,
        int(not candidate.failed),
        int(solvent),
        min(net_minimum, config.contingency_gauge),
        min(net_final, config.contingency_gauge),
        -liability,
        candidate.renewal_clears,
        candidate.score,
        -candidate.candidate_index,
    )
    return RankedCandidate(
        candidate, full_survival, net_minimum, net_final, solvent, rank
    )


def choose_candidate(
    candidates: Sequence[ProbeCandidate], config: ReserveBandConfig
) -> tuple[RankedCandidate, tuple[RankedCandidate, ...]]:
    if not candidates:
        raise ValueError("reserve-band comparison requires candidates")
    indices = [candidate.candidate_index for candidate in candidates]
    if len(set(indices)) != len(indices):
        raise ValueError("candidate indices must be unique")
    ranked = tuple(rank_candidate(candidate, config) for candidate in candidates)
    return max(ranked, key=lambda item: item.rank), ranked


def evaluate_manifest(value: Mapping[str, Any]) -> dict[str, object]:
    config_value, candidate_values = value.get("config"), value.get("candidates")
    if not isinstance(config_value, Mapping) or not isinstance(candidate_values, list):
        raise ValueError("input must contain config and candidate list")
    if any(not isinstance(item, Mapping) for item in candidate_values):
        raise ValueError("candidate entries must be objects")
    config = ReserveBandConfig.from_mapping(config_value)
    candidates = tuple(ProbeCandidate.from_mapping(item) for item in candidate_values)
    winner, ranked = choose_candidate(candidates, config)
    return {
        "version": VERSION,
        "development_only": True,
        "promotion_eligible": False,
        "config": config.manifest(),
        "rank_order": [
            item.candidate.candidate_index
            for item in sorted(ranked, key=lambda item: item.rank, reverse=True)
        ],
        "winner_candidate_index": winner.candidate.candidate_index,
        "ranked_candidates": [item.manifest() for item in ranked],
        "ranking_semantics": [
            "zero invalid actions",
            "survival ticks",
            "nonterminal full-horizon survival",
            "reserve-band solvency",
            "liability-adjusted minimum gauge capped at contingency band",
            "liability-adjusted final gauge capped at contingency band",
            "lower imminent visible rot liability",
            "renewal clears",
            "score",
            "lower candidate index tie-break",
        ],
    }


__all__ = [
    "ProbeCandidate",
    "RankedCandidate",
    "ReserveBandConfig",
    "VERSION",
    "choose_candidate",
    "evaluate_manifest",
    "rank_candidate",
]
