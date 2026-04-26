"""Разрешение путей к ресурсам приложения (модели и т.п.).

В dev-режиме корнем считается папка `segmentation_system/` (родитель `src/`).
При запуске под PyInstaller (`--onedir`) ресурсы лежат в `_internal/`,
куда указывает `sys._MEIPASS`.
"""
from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent.parent


def resource_path(rel: str) -> Path:
    return project_root() / rel
