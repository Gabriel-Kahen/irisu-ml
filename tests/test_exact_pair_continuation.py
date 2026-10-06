from __future__ import annotations

import pytest

from benchmarks import rl_exact_100k_continuation as continuation


def test_exact_pair_metadata_accepts_only_exact_training() -> None:
    continuation.validate_exact_pair_metadata(
        {
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "portable_checkpoint_loaded": False,
        }
    )
    with pytest.raises(ValueError, match="exact-only"):
        continuation.validate_exact_pair_metadata(
            {
                "physics_backend": "exact",
                "state_producing_backends": ["exact"],
                "portable_checkpoint_loaded": True,
            }
        )


def test_pair_trace_decode_does_not_require_portable_campaign() -> None:
    action = continuation.decode(None, (123 << 12) | (456 << 2) | 2)
    assert int(action.kind) == 2
    assert int(action.cursor_x) == 456
    assert int(action.cursor_y) == 123
