#!/usr/bin/env python3
"""Verify and merge v2 exact adaptive evaluator shards for promotion."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rl_exact_adaptive_checkpoint_merge import (  # noqa: E402
    FORMAT_V2,
    SHARD_FORMAT_V2,
    main,
)


if __name__ == "__main__":
    raise SystemExit(
        main(
            expected_shard_format=SHARD_FORMAT_V2,
            output_format=FORMAT_V2,
            merger_path=Path(__file__),
        )
    )
