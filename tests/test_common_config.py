from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.config import ParsedOverride, apply_overrides, load_yaml_config, parse_overrides


class TestCommonConfig(unittest.TestCase):
    def test_parse_overrides_coerce_types(self) -> None:
        parsed = parse_overrides(
            [
                "a.bool=true",
                "a.int=7",
                "a.float=2.5",
                "a.null=null",
                "a.text=hello",
            ]
        )

        self.assertEqual(parsed[0].value, True)
        self.assertEqual(parsed[1].value, 7)
        self.assertEqual(parsed[2].value, 2.5)
        self.assertIsNone(parsed[3].value)
        self.assertEqual(parsed[4].value, "hello")

    def test_apply_overrides_keeps_input_immutable(self) -> None:
        cfg = {"train": {"epochs": 10}, "names": ["a", "b"]}
        out = apply_overrides(
            cfg,
            [
                ParsedOverride(key="train.epochs", value=20),
                ParsedOverride(key="new.path.value", value="x"),
            ],
        )

        self.assertEqual(cfg["train"]["epochs"], 10)
        self.assertEqual(out["train"]["epochs"], 20)
        self.assertEqual(out["new"]["path"]["value"], "x")

    def test_load_yaml_config_empty_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "empty.yaml"
            cfg_file.write_text("", encoding="utf-8")
            cfg = load_yaml_config(cfg_file)
            self.assertEqual(cfg, {})

    def test_load_yaml_config_non_mapping_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "invalid.yaml"
            cfg_file.write_text("- a\n- b\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                _ = load_yaml_config(cfg_file)


if __name__ == "__main__":
    unittest.main()
