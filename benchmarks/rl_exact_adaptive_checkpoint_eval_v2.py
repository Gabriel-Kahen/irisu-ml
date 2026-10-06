#!/usr/bin/env python3
"""Exact adaptive evaluation with conservative fork-worker error recovery.

This v2 runner has the same CLI and score contract as v1.  If and only if an
exact counterfactual branch raises ``ExactWorkerError``, it verifies that the
live parent state hash is unchanged, restores the pre-proposal controller
state, records bound error evidence, and executes WAIT.  Other failures remain
fatal.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rl_exact_adaptive_checkpoint_eval import main  # noqa: E402


FORMAT = "irisu-exact-adaptive-learned-planner-eval-v2"


if __name__ == "__main__":
    raise SystemExit(
        main(
            report_format=FORMAT,
            runner_path=Path(__file__),
            recover_exact_branch_errors=True,
        )
    )
