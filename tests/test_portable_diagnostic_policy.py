from __future__ import annotations

import ast
import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/rl/portable-diagnostics-v1.toml"
PORTABLE_SOURCE = re.compile(
    r'physics_backend\s*=\s*["\']portable["\']|portable-build/libirisu_clone\.so'
)


def _registered_paths(section: dict[str, list[str]]) -> set[str]:
    paths = [path for group in section.values() for path in group]
    if len(paths) != len(set(paths)):
        raise AssertionError("portable diagnostic registry contains duplicate paths")
    return set(paths)


def _contains_portable_source(path: Path) -> bool:
    source = path.read_text()
    if PORTABLE_SOURCE.search(source):
        return True
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else ""
        )
        if name not in {"IrisuEnv", "PaddedVectorEnv"}:
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        if "physics_backend" not in keywords and None not in keywords:
            return True
    return False


class PortableDiagnosticPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = tomllib.loads(REGISTRY.read_text())

    def test_policy_unconditionally_bans_portable_promotion(self) -> None:
        policy = self.registry["policy"]
        self.assertEqual(policy["target_training_backend"], "exact")
        self.assertEqual(policy["legacy_runtime_role"], "legacy_portable_diagnostic")
        self.assertIs(policy["target_training_allowed"], False)
        self.assertIs(policy["promotion_eligible"], False)
        self.assertIs(policy["fresh_exact_lineage_required"], True)

    def test_every_portable_python_entrypoint_is_classified(self) -> None:
        registered = _registered_paths(self.registry["entrypoints"])
        discovered = {
            path.relative_to(ROOT).as_posix()
            for root in (ROOT / "benchmarks", ROOT / "python", ROOT / "tools")
            for path in root.rglob("*.py")
            if _contains_portable_source(path)
        }
        self.assertSetEqual(discovered, registered)
        for relative in registered:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_every_portable_training_config_is_classified(self) -> None:
        registered = _registered_paths(self.registry["configs"])
        discovered = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "configs/rl").rglob("*.toml")
            if path != REGISTRY
            if "portable" in path.read_text().lower()
        }
        self.assertSetEqual(discovered, registered)
        for relative in registered:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_legacy_configs_are_explicitly_non_promotable(self) -> None:
        for relative in self.registry["configs"]["legacy_portable"]:
            with self.subTest(config=relative):
                config = tomllib.loads((ROOT / relative).read_text())
                policy = config.get("experiment", config)
                self.assertEqual(
                    policy["runtime_role"], "legacy_portable_diagnostic"
                )
                self.assertIs(policy["target_training_allowed"], False)
                self.assertIs(policy["promotion_eligible"], False)
                self.assertIs(policy["fresh_exact_lineage_required"], True)

    def test_exact_primary_configs_make_portable_mode_diagnostic_only(self) -> None:
        for relative in self.registry["configs"][
            "exact_primary_with_portable_diagnostic"
        ]:
            with self.subTest(config=relative):
                config = tomllib.loads((ROOT / relative).read_text())
                if "simulator" in config:
                    simulator = config["simulator"]
                    self.assertEqual(simulator["training_backend"], "exact")
                    self.assertEqual(
                        simulator["portable_mode"],
                        "explicit_diagnostic_only_non_promotable",
                    )
                else:
                    backends = config["backends"]
                    self.assertIn("exact_environment_pool", backends)
                    self.assertIn("portable_environment_pool", backends)


if __name__ == "__main__":
    unittest.main()
