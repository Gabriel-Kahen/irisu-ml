from __future__ import annotations

import json

import pytest

from irisu_search import PLANNER_VERSION
from irisu_search.cli import main
from irisu_search.config import CampaignConfig
from irisu_search.coordinator import Coordinator, planner_identity
from irisu_search.neural import features, train_guide
from irisu_search.scheduler import next_job
from irisu_search.store import Store


def config(**changes):
    values = {"engine": ["fake-engine"], "oracle": "fake-oracle", "initial_seeds": 1,
              "candidates": 4, "survivors": 2, "control_seed": 7}
    values.update(changes)
    return CampaignConfig(**values)


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_run_rejects_nonfinite_or_nonpositive_budget_before_using_resources(seconds):
    # No store or engine is attached: validation must precede resource access.
    coordinator = Coordinator.__new__(Coordinator)
    with pytest.raises(ValueError, match="budget|finite|positive"):
        coordinator.run(seconds=seconds)


@pytest.mark.parametrize("changes", [
    {"initial_seeds": 0x100000000}, {"horizon": 100001}, {"bootstrap_ticks": 100001},
    {"training_epochs": 100001}, {"job_seconds": float("nan")},
    {"engine_timeout": float("inf")}, {"discover_new_seeds": 1}, {"train_every_jobs": -1},
    {"engine": "fake-engine"}, {"exploration_fraction": float("nan")},
])
def test_invalid_campaign_bounds_reject_during_initialization(changes):
    with pytest.raises((ValueError, TypeError)):
        config(**changes)


def test_probe_expansion_respects_engine_horizon_limit(tmp_path):
    cfg = config(horizon=100000, bootstrap_ticks=100000, maximum_ticks=200000,
                 finalist_seconds=1, minimum_seed_seconds=1,
                 lanes={"seed": .01, "combo": .01, "probe": .97, "wildcard": .01})
    summary = {"seed": 7, "tick": 1, "score": 0, "gauge": 1000, "qualifying": 0,
               "groups": [], "bodies": [], "terminated": False, "truncated": False}
    with Store(tmp_path) as store:
        store.ensure_seed(7, control=True)
        store.update_seed(7, attempts=3, search_seconds=1000)
        store.add_checkpoint(7, [0], summary)
        key = next_job(store, cfg)
        job = next(job for job in store.jobs() if job["id"] == key)
        assert job["lane"] == "probe"
        assert 0 < job["payload"]["horizon"] <= 100000


class IdentityEngine:
    identity = {"protocol": 1, "backend": "identity-only-fixture", "max_checkpoints": 64}

    def release(self, handles):
        assert handles == []


class IdentityVerifier:
    identity = {"backend": "identity-only-verifier"}


def guide_file(path, *, count=2):
    identity = planner_identity(IdentityEngine.identity, IdentityVerifier.identity)
    rows = [{"seed": seed, "features": features({}), "targets": {"extension": seed % 2},
             "planner_version": PLANNER_VERSION, "budget_seconds": 1, "censored": True}
            for seed in range(count)]
    model, report = train_guide(rows, epochs=1, hidden=3)
    model.save(path, {"engine_identity": identity})
    return identity, report


def test_config_model_cannot_bypass_whole_seed_validation(tmp_path):
    path = tmp_path / "single-seed.json"
    guide_file(path, count=1)
    with Store(tmp_path / "campaign") as store:
        with pytest.raises(ValueError, match="validation|held.out"):
            Coordinator(store, IdentityEngine(), config(model=str(path)), IdentityVerifier())


def test_config_model_is_hash_bound_before_subsequent_restart(tmp_path):
    path = tmp_path / "guide.json"
    guide_file(path)
    cfg = config(model=str(path))
    with Store(tmp_path / "campaign") as store:
        first = Coordinator(store, IdentityEngine(), cfg, IdentityVerifier())
        first.close()
        active = store.get_meta("active_model")
        assert active and len(active["sha256"]) == 64
        # Valid JSON, valid internal checksum, same provenance, different file.
        path.write_text(path.read_text() + "\n")
        with pytest.raises(ValueError, match="changed|sha|hash"):
            Coordinator(store, IdentityEngine(), cfg, IdentityVerifier())


def test_cli_does_not_train_or_create_output_for_missing_campaign(tmp_path, capsys):
    output = tmp_path / "guide.json"
    assert main(["train", str(tmp_path / "missing"), str(output), "--epochs", "1"]) == 1
    assert not output.exists()
    assert "not initialized" in capsys.readouterr().err


@pytest.mark.parametrize(("terminated", "truncated", "expected"), [
    (False, False, 1), (False, True, 1), (True, True, 1), (True, False, 0),
])
def test_cli_verify_requires_natural_termination(tmp_path, monkeypatch, capsys, terminated, truncated, expected):
    from irisu_search import cli

    cfg = config()
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        checkpoint_id = store.add_checkpoint(7, [0], {
            "seed": 7, "tick": 1, "score": 16, "canonical_score": 16,
            "terminated": terminated, "truncated": truncated,
        })
    verified = []

    class FixtureCoordinator:
        def _verify(self, key, force=False):
            verified.append((key, force))
            # A successful re-verification need not promote a new champion.
            return False

        def close(self):
            pass

    class FixtureEngine:
        def close(self):
            pass

    monkeypatch.setattr(cli, "open_coordinator", lambda *_: (FixtureEngine(), FixtureCoordinator()))
    assert cli.main(["verify", str(tmp_path), checkpoint_id]) == expected
    output = capsys.readouterr()
    if expected:
        assert "naturally terminated" in output.err
        assert not verified
        assert not output.out
    else:
        assert json.loads(output.out)["verified"] is True
        assert verified == [(checkpoint_id, True)]
