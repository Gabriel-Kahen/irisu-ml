from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3/development/exact-phase0-oracle-pilot-train4-20260813-001/oracle-pilot.json"
SPEC = importlib.util.spec_from_file_location(
    "rl_g5_solvency_trigger", ROOT / "benchmarks/rl_g5_solvency_trigger.py"
)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    artifact = json.loads(ARTIFACT.read_text())
    return {
        "schema": TRAINER.SCHEMA,
        "purpose": "g5-compute-trigger-development-training-only",
        "expected_base_checkpoint_sha256": artifact["base_checkpoint_sha256"],
        "expected_base_model_sha256": artifact["base_model_sha256"],
        "artifacts": [{
            "path": str(ARTIFACT),
            "sha256": _file_sha(ARTIFACT),
            "content_sha256": artifact["content_sha256"],
        }],
    }


def test_input_manifest_binds_oracle_and_builds_all_query_targets(tmp_path: Path) -> None:
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(_manifest()))
    _manifest_value, artifacts = TRAINER.load_inputs(path)
    features, targets, seeds, rows = TRAINER.dataset(artifacts)
    assert features.shape == (64, 712)
    assert len(set(seeds)) == 4
    assert len(rows) == 64
    assert int(sum(targets["delayed_disagreement"])) == 18


def test_input_manifest_rejects_wrong_base_lineage(tmp_path: Path) -> None:
    value = _manifest()
    value["expected_base_checkpoint_sha256"] = "0" * 64
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="lineage differs"):
        TRAINER.load_inputs(path)


def test_input_manifest_requires_direct_absolute_artifact_path(tmp_path: Path) -> None:
    value = _manifest()
    value["artifacts"][0]["path"] = str(ARTIFACT.relative_to(ROOT))
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="direct and absolute"):
        TRAINER.load_inputs(path)


def test_input_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "inputs.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        TRAINER.load_inputs(path)
