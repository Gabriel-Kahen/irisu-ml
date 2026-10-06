"""Hermetic import-owner fixtures for frozen G4 campaign security tests."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace


def assert_omitted_preload_captures_foreign_owner(runner, tmp_path: Path) -> None:
    foreign_root = tmp_path / "foreign-python"
    foreign_package = foreign_root / "irisu_env"
    foreign_package.mkdir(parents=True)
    foreign_init = foreign_package / "__init__.py"
    foreign_init.write_text('"""Foreign package owner fixture."""\n')
    barrier = tmp_path / "barrier_core.py"
    # Reproduce the historical barrier's path insertion without its old worktree.
    barrier.write_text(
        f"import sys\nsys.path.insert(0, {str(foreign_root)!r})\nimport irisu_env\n"
    )
    barrier_sha = hashlib.sha256(barrier.read_bytes()).hexdigest()
    script = f"""
import importlib, importlib.util, sys
from pathlib import Path
p = Path({str(runner.__file__)!r})
s = importlib.util.spec_from_file_location("_r3j_omitted_preload", p)
m = importlib.util.module_from_spec(s)
sys.modules[s.name] = m
s.loader.exec_module(m)
m.enforce_cpu0()
assert sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
assert sys.flags.safe_path and sys.pycache_prefix == str(m.PYTHON_CACHE_PREFIX)
# Test import ownership independently of the frozen campaign's runtime seal.
sys.path[:0] = [{str(runner.REPOSITORY / 'python')!r}, {sysconfig.get_path('purelib')!r}]
importlib.import_module("irisu_pointer.resolution_proposal_g4")
assert "irisu_env" not in sys.modules
m._load_module("barrier_core", Path({str(barrier)!r}), {barrier_sha!r})
print(Path(sys.modules["irisu_env"].__file__).resolve())
try:
    m._preload_main_irisu_env()
except RuntimeError as exc:
    assert "simulator module is foreign: irisu_env" in str(exc)
else:
    raise AssertionError("foreign simulator preload was accepted")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I", "-S", "-B", "-X",
            f"pycache_prefix={runner.PYTHON_CACHE_PREFIX}",
            "-c", script,
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            **{name: "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
            )},
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(foreign_init)


def install_verified_module_closure(verifier, tmp_path: Path, monkeypatch):
    """Give the real closure verifier complete files with independently bound hashes."""
    extras = ("barrier_core", "campaign_metrics", "campaign")
    expected = {}
    for name in (*verifier.TRANSITIVE_MODULES, *extras, "r3j_live_tau2_lease"):
        path = tmp_path / "owners" / f"{name}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"# G4 closure fixture: {name}\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected[name] = (path, digest)
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(path)))
    monkeypatch.setattr(verifier, "TRANSITIVE_MODULES", {
        name: expected[name] for name in verifier.TRANSITIVE_MODULES
    })
    monkeypatch.setattr(verifier, "B_PATHS", {name: expected[name][0] for name in extras})
    monkeypatch.setattr(verifier, "B_SHA256", {name: expected[name][1] for name in extras})
    live_path, live_sha = expected["r3j_live_tau2_lease"]
    monkeypatch.setattr(verifier, "LIVE_TAU2_SOURCE", live_path)
    # Hold this completed namespace; parallel tests may mutate shared /tmp ancestors.
    with verifier.VerificationLease(tmp_path):
        assert verifier.verify_module_closure(live_sha) == {
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in expected.items()
        }
    return expected, live_sha
