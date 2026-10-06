from __future__ import annotations

import struct
import unittest
from pathlib import Path

from irisu_env import Action

from benchmarks.rl_exact_trace_distillation import (
    DEFAULT_CONFIG,
    MODEL_HEADER,
    MODEL_MAGIC,
    MODEL_RECORD,
    ExactTraceTablePolicy,
    decode_action,
    encode_action,
    load_config,
    public_observation_sha256,
    stable_acceptance_report,
)


class ExactTraceDistillationTests(unittest.TestCase):
    def test_config_is_exact_checkpoint_free_and_has_no_fallback(self) -> None:
        value = load_config(DEFAULT_CONFIG)
        self.assertEqual(value["physics_backend"], "exact")
        self.assertEqual(value["state_producing_backends"], ["exact"])
        self.assertEqual(value["checkpoint_dependencies"], [])
        self.assertEqual(value["fallback"], "none")

    def test_public_digest_is_canonical_and_state_sensitive(self) -> None:
        first = {"tick": 2, "bodies": [{"x": 1.25, "id": 7}], "score": 3}
        reordered = {"score": 3, "bodies": [{"id": 7, "x": 1.25}], "tick": 2}
        changed = {**first, "score": 4}
        self.assertEqual(
            public_observation_sha256(first), public_observation_sha256(reordered)
        )
        self.assertNotEqual(
            public_observation_sha256(first), public_observation_sha256(changed)
        )

    def test_action_words_round_trip_without_campaign_code(self) -> None:
        for action in (
            Action.wait(1),
            Action.weak(130, 120),
            Action.strong(449, 369),
            Action.both(300, 200),
        ):
            self.assertEqual(encode_action(decode_action(encode_action(action))), encode_action(action))
        with self.assertRaises(ValueError):
            decode_action(4)

    def test_table_policy_is_closed_loop_and_fails_on_unseen_state(self) -> None:
        observation = {"tick": 0, "score": 0, "bodies": []}
        word = encode_action(Action.weak(200, 150))
        payload = MODEL_HEADER.pack(MODEL_MAGIC, 41, 1) + MODEL_RECORD.pack(
            0, public_observation_sha256(observation), word
        )
        policy = ExactTraceTablePolicy(payload)
        self.assertEqual(policy.seed, 41)
        self.assertEqual(encode_action(policy.predict(observation)), word)
        self.assertTrue(policy.complete)
        with self.assertRaises(RuntimeError):
            policy.predict(observation)

        policy = ExactTraceTablePolicy(payload)
        with self.assertRaisesRegex(RuntimeError, "unseen public state"):
            policy.predict({**observation, "score": 1})

    def test_acceptance_evidence_discards_only_ephemeral_worker_pid(self) -> None:
        report = {"clone_build": {"worker_pid": 123, "physics_backend": "exact"}}
        stable = stable_acceptance_report(report)
        self.assertNotIn("worker_pid", stable["clone_build"])
        self.assertIs(stable["clone_build"]["worker_process_attested"], True)
        self.assertIn("worker_pid", report["clone_build"])

    def test_source_has_no_old_campaign_or_learned_checkpoint_dependency(self) -> None:
        source = __import__(
            "benchmarks.rl_exact_trace_distillation", fromlist=["__file__"]
        ).__file__
        assert source is not None
        text = Path(source).read_text()
        self.assertNotIn("rl_r3k", text)
        self.assertNotIn("POLICY_FACTORY", text)
        self.assertNotIn("torch", text)
        self.assertNotIn(".pt", text)


if __name__ == "__main__":
    unittest.main()
