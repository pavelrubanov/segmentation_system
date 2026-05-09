"""core/paths.py: разрешение путей в dev и frozen-режиме."""
import sys
from pathlib import Path

from core.paths import project_root, resource_path


class TestProjectRoot:
    def test_dev_mode_contains_src_dir(self):
        # В dev-режиме project_root — родитель src/, поэтому src/ обязан существовать.
        assert (project_root() / "src").exists()

    def test_frozen_mode_uses_meipass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert project_root() == tmp_path


class TestResourcePath:
    def test_joins_with_root(self):
        assert resource_path("models/foo.pt") == project_root() / "models" / "foo.pt"

    def test_returns_path_object(self):
        assert isinstance(resource_path("anything"), Path)

    def test_handles_nested_path(self):
        p = resource_path("a/b/c")
        assert p.parts[-3:] == ("a", "b", "c")
