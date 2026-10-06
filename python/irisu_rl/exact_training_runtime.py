"""Fail-closed construction and promotion contract for exact RL runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from irisu_env import IrisuEnv, PaddedVectorEnv

from .runtime_identity import attest_simulator_runtime


DEFAULT_IDENTITY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "rl"
    / "runtime"
    / "exact-worker-2026-07-21.json"
)
_IDENTITY_KEYS = frozenset(
    {
        "version",
        "worker_path_policy",
        "worker_sha256",
        "exact_library_sha256",
        "protocol_version",
        "body_capacity",
        "pointer_bits",
        "backend",
        "evidence",
    }
)
_BACKEND_KEYS = frozenset(
    {
        "backend",
        "physics_backend",
        "runtime_backend",
        "training_backend",
        "evaluation_backend",
    }
)
_LINEAGE_KEYS = frozenset(
    {
        "lineage",
        "model_lineage",
        "checkpoint_lineage",
        "parent_lineage",
        "warm_start_lineage",
    }
)
_ZERO_SHA256 = "0" * 64
_T = TypeVar("_T", IrisuEnv, PaddedVectorEnv)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != _ZERO_SHA256
        and all(character in "0123456789abcdef" for character in value)
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_file(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    try:
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot read {label}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(current)
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while it was read")
    return payload, current


@dataclass(frozen=True, slots=True)
class ExactTrainingIdentity:
    version: str
    worker_path_policy: str
    worker_sha256: str
    exact_library_sha256: str
    protocol_version: int
    body_capacity: int
    pointer_bits: int
    backend: str
    evidence: str
    config_path: str
    config_sha256: str

    @classmethod
    def load(
        cls, path: str | os.PathLike[str] = DEFAULT_IDENTITY_PATH
    ) -> "ExactTrainingIdentity":
        supplied = Path(path).expanduser()
        if not supplied.is_absolute():
            raise ValueError("exact runtime identity config path must be absolute")
        payload, _ = _read_stable_file(supplied, "exact runtime identity config")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("exact runtime identity config is not valid JSON") from exc
        if not isinstance(decoded, dict) or set(decoded) != _IDENTITY_KEYS:
            raise ValueError("exact runtime identity config has an unexpected schema")
        integer_keys = ("protocol_version", "body_capacity", "pointer_bits")
        if (
            decoded["version"] != "exact-runtime-identity-v1"
            or decoded["worker_path_policy"] != "absolute-path-required-at-runtime"
            or not _is_sha256(decoded["worker_sha256"])
            or not _is_sha256(decoded["exact_library_sha256"])
            or any(
                isinstance(decoded[key], bool)
                or not isinstance(decoded[key], int)
                or decoded[key] <= 0
                for key in integer_keys
            )
            or not isinstance(decoded["backend"], str)
            or not decoded["backend"].startswith("exact-")
            or not isinstance(decoded["evidence"], str)
            or not decoded["evidence"]
        ):
            raise ValueError("exact runtime identity config is malformed")
        return cls(
            **decoded,
            config_path=str(supplied.resolve(strict=True)),
            config_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "worker_path_policy": self.worker_path_policy,
            "worker_sha256": self.worker_sha256,
            "exact_library_sha256": self.exact_library_sha256,
            "protocol_version": self.protocol_version,
            "body_capacity": self.body_capacity,
            "pointer_bits": self.pointer_bits,
            "backend": self.backend,
            "evidence": self.evidence,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
        }


def _validated_worker(path_value: str | os.PathLike[str], expected_sha256: str) -> Path:
    supplied = Path(path_value).expanduser()
    if not supplied.is_absolute():
        raise ValueError("exact worker path must be explicit and absolute")
    payload, metadata = _read_stable_file(supplied, "exact worker executable")
    if not metadata.st_mode & 0o111:
        raise ValueError("exact worker executable is not executable")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("exact worker executable SHA-256 mismatch")
    return supplied.resolve(strict=True)


def _backend_claims(
    value: object, path: str = "metadata"
) -> tuple[tuple[str, str], ...]:
    claims: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("promotion metadata keys must be strings")
            child_path = f"{path}.{key}"
            if key in _BACKEND_KEYS:
                if not isinstance(child, str) or not child.strip():
                    raise TypeError(f"{child_path} must be a nonempty string")
                claims.append((child_path, child.strip().lower().replace("_", "-")))
            claims.extend(_backend_claims(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            claims.extend(_backend_claims(child, f"{path}[{index}]"))
    return tuple(claims)


def _portable_lineage_claims(
    value: object, path: str = "metadata", *, in_lineage: bool = False
) -> tuple[tuple[str, str], ...]:
    claims: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("promotion metadata keys must be strings")
            child_path = f"{path}.{key}"
            child_in_lineage = in_lineage or key in _LINEAGE_KEYS
            claims.extend(
                _portable_lineage_claims(
                    child, child_path, in_lineage=child_in_lineage
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            claims.extend(
                _portable_lineage_claims(
                    child, f"{path}[{index}]", in_lineage=in_lineage
                )
            )
    elif in_lineage and isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        if "portable" in normalized or "mixed" in normalized:
            claims.append((path, normalized))
    return tuple(claims)


def validate_exact_promotion_metadata(metadata: Mapping[str, object]) -> None:
    """Reject promotion records that are portable, ambiguous, or internally mixed."""

    if not isinstance(metadata, Mapping):
        raise TypeError("promotion metadata must be a mapping")
    if metadata.get("physics_backend") != "exact":
        raise ValueError("promotion metadata must declare physics_backend='exact'")
    invalid = [
        (path, backend)
        for path, backend in _backend_claims(metadata)
        if backend != "exact" and not backend.startswith("exact-")
    ]
    if invalid:
        raise ValueError(f"promotion metadata contains a non-exact backend: {invalid}")
    portable_lineage = _portable_lineage_claims(metadata)
    if portable_lineage:
        raise ValueError(
            "promotion metadata contains portable or mixed model lineage: "
            f"{portable_lineage}"
        )


class ExactTrainingSession(Generic[_T]):
    """An environment that was withheld until exact-runtime attestation passed."""

    def __init__(self, environment: _T, provenance: Mapping[str, object]) -> None:
        self.environment = environment
        self._provenance_json = _canonical_json(provenance)

    @property
    def provenance_manifest(self) -> dict[str, object]:
        return json.loads(self._provenance_json)

    def promotion_metadata(
        self, metadata: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        supplied = {} if metadata is None else dict(metadata)
        if "physics_backend" in supplied and supplied["physics_backend"] != "exact":
            raise ValueError("portable metadata cannot be promoted by an exact runtime")
        if "exact_runtime" in supplied:
            raise ValueError("exact_runtime promotion metadata is contract-owned")
        result = {
            **supplied,
            "physics_backend": "exact",
            "exact_runtime": self.provenance_manifest,
        }
        validate_exact_promotion_metadata(result)
        return result

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> "ExactTrainingSession[_T]":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ExactTrainingRuntime:
    """Factory pinned to the accepted exact worker and mapped legacy library."""

    def __init__(self, worker_path: str | os.PathLike[str]) -> None:
        self.identity = ExactTrainingIdentity.load(DEFAULT_IDENTITY_PATH)
        self.worker_path = _validated_worker(worker_path, self.identity.worker_sha256)

    def _attest(self, environment: _T, expected_lanes: int) -> ExactTrainingSession[_T]:
        attestation = attest_simulator_runtime(environment)
        info = attestation.build_info
        expected_info = {
            "protocol_version": self.identity.protocol_version,
            "body_capacity": self.identity.body_capacity,
            "pointer_bits": self.identity.pointer_bits,
            "worker_backend": self.identity.backend,
        }
        mismatches = {
            key: (info.get(key), expected)
            for key, expected in expected_info.items()
            if info.get(key) != expected
        }
        if (
            getattr(environment, "physics_backend", None) != "exact"
            or attestation.backend != "exact"
            or attestation.verified_lanes != expected_lanes
            or attestation.runtime_artifact_sha256 != self.identity.worker_sha256
            or attestation.exact_library_sha256 != self.identity.exact_library_sha256
            or any(
                Path(path) != self.worker_path
                for path in attestation.runtime_artifact_paths
            )
            or info.get("exact_library_runtime_verified") is not True
            or info.get("exact_call_targets_runtime_verified") is not True
            or mismatches
        ):
            raise RuntimeError(
                f"exact training runtime attestation failed: {mismatches}"
            )
        runner_manifest = environment.runner_identity_manifest()
        if runner_manifest.get("physics_backend") != "exact":
            raise RuntimeError("exact training runner identity is not exact")
        provenance = {
            "version": "exact-training-runtime-provenance-v1",
            "physics_backend": "exact",
            "identity": self.identity.manifest(),
            "runtime_attestation_sha256": attestation.sha256,
            "runtime_attestation": attestation.evidence_manifest(),
            "runner_identity": runner_manifest,
        }
        return ExactTrainingSession(environment, provenance)

    def open_env(
        self,
        *,
        simulation_config: Mapping[str, Any] | None = None,
        render_mode: str | None = None,
        diagnostic_hashes: bool = False,
    ) -> ExactTrainingSession[IrisuEnv]:
        self.worker_path = _validated_worker(
            self.worker_path, self.identity.worker_sha256
        )
        environment = IrisuEnv(
            config=simulation_config,
            render_mode=render_mode,
            diagnostic_hashes=diagnostic_hashes,
            physics_backend="exact",
            worker_path=self.worker_path,
        )
        try:
            return self._attest(environment, 1)
        except BaseException:
            environment.close()
            raise

    def open_vector(
        self,
        num_envs: int,
        *,
        simulation_config: Mapping[str, Any] | None = None,
        workers: int | None = None,
    ) -> ExactTrainingSession[PaddedVectorEnv]:
        self.worker_path = _validated_worker(
            self.worker_path, self.identity.worker_sha256
        )
        environment = PaddedVectorEnv(
            num_envs,
            config=simulation_config,
            workers=workers,
            physics_backend="exact",
            worker_path=self.worker_path,
        )
        try:
            return self._attest(environment, num_envs)
        except BaseException:
            environment.close()
            raise
