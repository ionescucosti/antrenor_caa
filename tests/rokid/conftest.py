"""Pytest fixtures for the Rokid backend. Helpers live in helpers.py."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from helpers import make_settings  # noqa: E402  (tests/rokid is on sys.path, see helpers.py)
from rokid_backend.llm.mock import MockLLM
from rokid_backend.main import create_app


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def client(tmp_path, mock_llm):
    app = create_app(make_settings(tmp_path), llm=mock_llm)
    with TestClient(app) as c:
        c.app = app
        yield c


@pytest.fixture
def general_client(tmp_path, mock_llm):
    app = create_app(make_settings(tmp_path, allow_general_knowledge=True), llm=mock_llm)
    with TestClient(app) as c:
        c.app = app
        yield c
