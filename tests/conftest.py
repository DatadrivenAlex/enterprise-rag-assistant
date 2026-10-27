"""Offline test defaults. No API key and no model download."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from enterprise_rag.config import Settings
from tests.helpers import make_settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)
