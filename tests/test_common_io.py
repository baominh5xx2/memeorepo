from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from damp_es.common.io import list_images


class TestCommonIO(unittest.TestCase):
    def test_list_images_skips_hidden_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").write_bytes(b"x")
            (root / ".b.png").write_bytes(b"x")
            (root / "._c.png").write_bytes(b"x")
            (root / "note.txt").write_text("x", encoding="utf-8")

            files = list_images(root)
            self.assertEqual([p.name for p in files], ["a.png"])


if __name__ == "__main__":
    unittest.main()
