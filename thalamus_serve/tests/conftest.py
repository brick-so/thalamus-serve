from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from thalamus_serve.core.app import app
from thalamus_serve.testing import TEST_API_KEY


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("THALAMUS_API_KEY", TEST_API_KEY)
    with TestClient(app) as c:
        yield c
