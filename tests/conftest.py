"""Конфигурация pytest: добавляем src/ в sys.path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
