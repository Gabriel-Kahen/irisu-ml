"""Keep the restored planner traceable to the actual October source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def functions(path):
    tree = ast.parse(path.read_text())
    # Relative imports are the only internal package-name adaptation.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            node.module = node.module.removeprefix("rl_combo_")
            node.level = 0
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def test_recovered_source_hashes_and_retained_function_bodies():
    originals = ROOT / "tools/historical/original"
    manifest = json.loads((originals.parent / "recovery-manifest.json").read_text())
    assert manifest["historical_search_sha256"] == (
        "a800d330a3e53c37eff711d60bcf406fe644f956ac3022d0258d05344e90b875")
    exceptions = {"event_options": {"_step"}, "growth_search": {"replay_events"},
                  "diverse_search": {"defaults"}}
    for name, receipt in manifest["files"].items():
        original = originals / name
        assert hashlib.sha256(original.read_bytes()).hexdigest() == receipt["sha256"]
        module = name.removeprefix("rl_combo_").removesuffix(".py")
        old = functions(original)
        current = functions(ROOT / "python/irisu_search/historical" / f"{module}.py")
        for function, body in current.items():
            if function not in exceptions.get(module, set()):
                assert old[function] == body, (module, function)
