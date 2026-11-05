from __future__ import annotations

import json
import struct
import sys

import pytest

from irisu_search.verification import NativeVerifier, read_replay, replay_bytes


def checkpoint():
    return {"seed": 77, "words": [0, 0, 0, 0], "summary": {
        "tick": 4, "score": 598, "level": 99, "highest_chain": 9,
        "terminated": True, "truncated": False,
        "canonical_score": 500, "canonical_level": 99, "canonical_chain": 9,
    }}


def oracle(tmp_path, mode="valid"):
    script = tmp_path / ("oracle_" + mode)
    script.write_text(f'''#!{sys.executable}
import json, struct, sys
data = open(sys.argv[1], 'rb').read()
seed, level, score, chain, mode = struct.unpack_from('<I4i', data)
ticks = (len(data) - 52) // 4
behavior = {mode!r}
if behavior == 'exit':
    print('explicit failed replay', file=sys.stderr)
    sys.exit(2)
rows = ticks + 1 - int(behavior == 'short')
for tick in range(rows):
    sys.stdout.buffer.write(struct.pack('<12Q', tick,0,0,0,0,0,0,score,0,level,chain,score))
report = {{'backend': 'native-msvc-cpu-oracle', 'tick': ticks, 'steps': ticks,
          'canonical_score': score, 'live_score': score + 98, 'level': level,
          'highest_chain': chain}}
if behavior == 'mismatch': report['canonical_score'] += 1
if behavior == 'wrongbackend': report['backend'] = 'approximate'
if behavior == 'live_delta':
    report['level'] += 1
    report['highest_chain'] += 1
if behavior != 'missing':
    print('diagnostic preamble', file=sys.stderr)
    print(json.dumps(report), file=sys.stderr)
''')
    script.chmod(0o700)
    return script


def test_replay_uses_canonical_metadata_and_roundtrips(tmp_path):
    cp = checkpoint()
    path = tmp_path / "candidate.rpy"
    path.write_bytes(replay_bytes(cp["seed"], cp["words"], cp["summary"]))
    seed, words, header = read_replay(path)
    assert seed == 77 and words == cp["words"]
    assert header == {"canonical_score": 500, "canonical_level": 99, "canonical_chain": 9}
    assert struct.unpack_from("<i", path.read_bytes(), 8)[0] != cp["summary"]["score"]


def test_verifier_executes_independent_process_and_binds_receipt(tmp_path):
    cp = checkpoint()
    verifier = NativeVerifier(oracle(tmp_path))
    receipt = verifier.verify(cp)
    assert receipt["verified"] is True
    assert receipt["seed"] == cp["seed"]
    assert receipt["canonical_score"] == 500
    assert receipt["canonical_level"] == 99
    assert receipt["canonical_chain"] == 9
    assert len(receipt["replay_sha256"]) == 64
    assert receipt["runtime"] == verifier.identity


def test_live_final_update_metadata_can_differ_from_recorded_snapshot(tmp_path):
    cp = checkpoint()
    cp["summary"]["level"] = 100
    cp["summary"]["highest_chain"] = 10
    receipt = NativeVerifier(oracle(tmp_path, "live_delta")).verify(cp)
    assert receipt["canonical_level"] == 99
    assert receipt["canonical_chain"] == 9
    assert receipt["oracle_report"]["level"] == 100
    assert receipt["oracle_report"]["highest_chain"] == 10


@pytest.mark.parametrize("mode", ["short", "missing", "mismatch", "wrongbackend", "exit"])
def test_invalid_or_incomplete_oracle_results_reject(tmp_path, mode):
    with pytest.raises(RuntimeError):
        NativeVerifier(oracle(tmp_path, mode)).verify(checkpoint())


def test_runtime_mutation_after_binding_rejects(tmp_path):
    path = oracle(tmp_path)
    verifier = NativeVerifier(path)
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        verifier.verify(checkpoint())


def test_export_rejects_capped_postterminal_and_invalid_words():
    cp = checkpoint()
    with pytest.raises(ValueError):
        replay_bytes(cp["seed"], cp["words"] + [0], cp["summary"])
    cp["summary"]["truncated"] = True
    with pytest.raises(ValueError):
        replay_bytes(cp["seed"], cp["words"], cp["summary"])
    cp = checkpoint()
    cp["words"][-1] = 1 << 21
    with pytest.raises(ValueError):
        replay_bytes(cp["seed"], cp["words"], cp["summary"])


def test_reader_rejects_non_normal_mode_and_illegal_word(tmp_path):
    cp = checkpoint()
    data = bytearray(replay_bytes(cp["seed"], cp["words"], cp["summary"]))
    path = tmp_path / "bad.rpy"
    struct.pack_into("<i", data, 16, 1)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="normal-mode"):
        read_replay(path)
    struct.pack_into("<i", data, 16, 0)
    struct.pack_into("<I", data, 52, 1 << 21)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="input word"):
        read_replay(path)
