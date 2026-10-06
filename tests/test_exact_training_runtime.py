from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from irisu_rl.exact_training_runtime import (
    DEFAULT_IDENTITY_PATH,
    ExactTrainingIdentity,
    ExactTrainingRuntime,
    validate_exact_promotion_metadata,
)
from irisu_rl.runtime_identity import SimulatorRuntimeAttestation


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _FakeEnvironment:
    physics_backend = "exact"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False

    def runner_identity_manifest(self) -> dict[str, object]:
        return {"version": "test-runner-v1", "physics_backend": "exact"}

    def close(self) -> None:
        self.closed = True


class ExactTrainingRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker_bytes = b"exact-test-worker"
        self.library_bytes = b"exact-test-library"
        self.worker = self.root / "worker"
        self.library = self.root / "legacy.dll"
        self.worker.write_bytes(self.worker_bytes)
        self.worker.chmod(0o755)
        self.library.write_bytes(self.library_bytes)
        self.identity_path = self.root / "identity.json"
        self.identity_path.write_text(
            json.dumps(
                {
                    "version": "exact-runtime-identity-v1",
                    "worker_path_policy": "absolute-path-required-at-runtime",
                    "worker_sha256": _sha256(self.worker_bytes),
                    "exact_library_sha256": _sha256(self.library_bytes),
                    "protocol_version": 1,
                    "body_capacity": 196,
                    "pointer_bits": 32,
                    "backend": "exact-msvc9-r58-multiworld-forward",
                    "evidence": "test-evidence.json",
                }
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _attestation(
        self, lanes: int = 1, *, library_sha256: str | None = None
    ) -> SimulatorRuntimeAttestation:
        library_sha = library_sha256 or _sha256(self.library_bytes)
        build_info = {
            "physics_backend": "exact-msvc9-r58-worker",
            "snapshot_schema": 0x45580001,
            "protocol_version": 1,
            "body_capacity": 196,
            "pointer_bits": 32,
            "worker_backend": "exact-msvc9-r58-multiworld-forward",
            "worker_executable_sha256": _sha256(self.worker_bytes),
            "exact_library_sha256": library_sha,
            "exact_library_runtime_verified": True,
            "exact_call_targets_runtime_verified": True,
        }
        return SimulatorRuntimeAttestation(
            backend="exact",
            snapshot_schema=0x45580001,
            build_info_json=json.dumps(
                build_info, sort_keys=True, separators=(",", ":")
            ),
            runtime_artifact_kind="worker-executable",
            runtime_artifact_sha256=_sha256(self.worker_bytes),
            runtime_artifact_bytes=len(self.worker_bytes),
            exact_library_sha256=library_sha,
            exact_library_bytes=len(self.library_bytes),
            runtime_artifact_paths=(str(self.worker),) * lanes,
            exact_library_paths=(str(self.library),) * lanes,
        )

    def _runtime(self) -> ExactTrainingRuntime:
        with patch(
            "irisu_rl.exact_training_runtime.DEFAULT_IDENTITY_PATH",
            self.identity_path,
        ):
            return ExactTrainingRuntime(self.worker)

    def test_repository_identity_config_is_the_pinned_exact_runtime(self) -> None:
        identity = ExactTrainingIdentity.load()

        self.assertEqual(Path(identity.config_path), DEFAULT_IDENTITY_PATH)
        self.assertEqual(
            identity.worker_sha256,
            "4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261",
        )
        self.assertEqual(
            identity.exact_library_sha256,
            "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5",
        )

    def test_worker_must_be_absolute_regular_nonsymlink_executable_and_match(
        self,
    ) -> None:
        relative = Path("worker")
        with patch(
            "irisu_rl.exact_training_runtime.DEFAULT_IDENTITY_PATH",
            self.identity_path,
        ), self.assertRaisesRegex(ValueError, "absolute"):
            ExactTrainingRuntime(relative)

        self.worker.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "not executable"):
            self._runtime()
        self.worker.chmod(0o755)

        link = self.root / "worker-link"
        link.symlink_to(self.worker)
        with patch(
            "irisu_rl.exact_training_runtime.DEFAULT_IDENTITY_PATH",
            self.identity_path,
        ), self.assertRaisesRegex(ValueError, "symlink"):
            ExactTrainingRuntime(link)

        self.worker.write_bytes(b"wrong-worker")
        self.worker.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            self._runtime()

    def test_open_env_withholds_runtime_until_exact_attestation_and_emits_provenance(
        self,
    ) -> None:
        runtime = self._runtime()
        environment = _FakeEnvironment()
        attestation = self._attestation()

        with patch(
            "irisu_rl.exact_training_runtime.IrisuEnv", return_value=environment
        ) as constructor, patch(
            "irisu_rl.exact_training_runtime.attest_simulator_runtime",
            return_value=attestation,
        ):
            session = runtime.open_env(simulation_config={"gauge_max": 10})

        self.assertIs(session.environment, environment)
        self.assertEqual(constructor.call_args.kwargs["physics_backend"], "exact")
        self.assertEqual(constructor.call_args.kwargs["worker_path"], self.worker)
        manifest = session.provenance_manifest
        self.assertEqual(manifest["physics_backend"], "exact")
        self.assertEqual(
            manifest["runtime_attestation"]["verified_lanes"], 1
        )
        self.assertTrue(
            manifest["runtime_attestation"]["build_info"][
                "exact_call_targets_runtime_verified"
            ]
        )
        promotion = session.promotion_metadata({"model_sha256": "a" * 64})
        self.assertEqual(promotion["physics_backend"], "exact")
        self.assertEqual(promotion["exact_runtime"], manifest)
        with self.assertRaisesRegex(ValueError, "portable"):
            session.promotion_metadata({"physics_backend": "portable"})

    def test_vector_uses_only_exact_lanes_and_closes_on_attestation_failure(
        self,
    ) -> None:
        runtime = self._runtime()
        good = _FakeEnvironment()
        with patch(
            "irisu_rl.exact_training_runtime.PaddedVectorEnv", return_value=good
        ) as constructor, patch(
            "irisu_rl.exact_training_runtime.attest_simulator_runtime",
            return_value=self._attestation(3),
        ):
            session = runtime.open_vector(3, workers=2)

        self.assertEqual(constructor.call_args.args, (3,))
        self.assertEqual(constructor.call_args.kwargs["physics_backend"], "exact")
        self.assertEqual(
            session.provenance_manifest["runtime_attestation"]["verified_lanes"], 3
        )

        rejected = _FakeEnvironment()
        with patch(
            "irisu_rl.exact_training_runtime.PaddedVectorEnv", return_value=rejected
        ), patch(
            "irisu_rl.exact_training_runtime.attest_simulator_runtime",
            side_effect=RuntimeError("mapped call targets were not verified"),
        ):
            with self.assertRaisesRegex(RuntimeError, "call targets"):
                runtime.open_vector(2)
        self.assertTrue(rejected.closed)

    def test_wrong_mapped_library_is_rejected_and_portable_metadata_is_never_promotable(
        self,
    ) -> None:
        runtime = self._runtime()
        environment = _FakeEnvironment()
        wrong_library = _sha256(b"wrong-library")
        with patch(
            "irisu_rl.exact_training_runtime.IrisuEnv", return_value=environment
        ), patch(
            "irisu_rl.exact_training_runtime.attest_simulator_runtime",
            return_value=self._attestation(library_sha256=wrong_library),
        ):
            with self.assertRaisesRegex(RuntimeError, "attestation failed"):
                runtime.open_env()
        self.assertTrue(environment.closed)

        for metadata in (
            {"physics_backend": "portable"},
            {
                "physics_backend": "exact",
                "evaluation": {"backend": "portable-gnu-r58"},
            },
            {"physics_backend": "exact", "training_backend": "unknown"},
            {
                "physics_backend": "exact",
                "model_lineage": "portable-frozen-v5-warm-start",
            },
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_exact_promotion_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
