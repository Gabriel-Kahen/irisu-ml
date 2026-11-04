"""Check signed objectives and stable global ranks without CUDA/native libraries."""
from pathlib import Path
import runpy
import struct

import pytest


REFERENCE = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "tools/cuda-native/gpu_search_native_reference.py"
))


@pytest.mark.parametrize("objective", range(21, 32))
def test_gpu_reference_signed_objectives_and_failure_priority(objective):
    rows = [[0] * 32 for _ in range(6)]
    for row, value in zip(rows, [2**63 - 1, -(2**63), -1, 5, 5, 0]):
        row[objective] = value
    rows[0][17] = 1
    assert REFERENCE["ranking"](rows) == [3, 4, 5, 2, 1, 0]


def test_gpu_reference_packet_retains_rows_and_original_id_ties():
    rows = [[i] + [0] * 31 for i in range(7)]
    rows[0][21] = -1
    data = REFERENCE["packet"](rows)
    assert struct.unpack_from("<4I", data) == (0x32535249, 2, 7, 256)
    assert len(data) == 16 + 7 * 260
    assert [REFERENCE["RESULT"].unpack_from(data, 16 + i * 256)
            for i in range(7)] == [tuple(row) for row in rows]
    assert struct.unpack_from("<7I", data, 16 + 7 * 256) == (1, 2, 3, 4, 5, 6, 0)
