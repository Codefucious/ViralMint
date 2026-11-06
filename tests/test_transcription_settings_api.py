# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025-2026 ViralMint Contributors
"""The transcription model setting and the live model list, over HTTP."""
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from backend.core.exceptions import ExternalTranscriptionError
from backend.main import create_app
from backend.messaging import manager as messaging_manager
from backend.services import openrouter_transcription as stt


@pytest.fixture(scope="module")
def client():
    async def _noop(*a, **k):
        return None

    messaging_manager.messaging.start_all = _noop  # type: ignore[assignment]
    messaging_manager.messaging.stop_all = _noop  # type: ignore[assignment]
    with TestClient(create_app()) as c:
        yield c


def test_setting_saves_and_clears(client):
    slug = "openai/whisper-large-v3"
    saved = client.post("/api/settings", json={"transcription_model": slug})
    assert saved.status_code == 200
    assert saved.json()["transcription_model"] == slug
    assert client.get("/api/settings").json()["transcription_model"] == slug

    cleared = client.post("/api/settings", json={"transcription_model": ""})
    assert cleared.json()["transcription_model"] is None


def test_saving_another_setting_leaves_the_model_alone(client):
    slug = "deepgram/nova-3"
    client.post("/api/settings", json={"transcription_model": slug})
    other = client.post("/api/settings", json={"music_enabled": False})
    assert other.json()["transcription_model"] == slug
    client.post("/api/settings", json={"transcription_model": ""})


@pytest.mark.parametrize("bad", ["whisper-1", "a/b c", "a/" + "x" * 100, "../../etc/passwd"])
def test_a_malformed_slug_saves_nothing(client, bad):
    response = client.post("/api/settings", json={"transcription_model": bad})
    assert response.status_code == 400
    assert client.get("/api/settings").json()["transcription_model"] is None


def test_key_availability_is_a_boolean_never_the_key(client, monkeypatch):
    monkeypatch.setattr("backend.core.ai_provider.settings.OPENROUTER_API_KEY", "sk-or-secret")
    response = client.get("/api/settings")
    assert response.json()["openrouter_key_available"] is True
    assert "sk-or-secret" not in response.text


def test_model_list_is_not_swallowed_by_the_generic_config_route(client, monkeypatch):
    models = [{"id": "a/alpha", "name": "Alpha"}]
    monkeypatch.setattr(stt, "list_transcription_models", AsyncMock(return_value=(models, False)))
    response = client.get("/api/config/transcription_models")
    assert response.status_code == 200
    assert response.json() == {"models": models, "stale": False}


def test_model_list_failure_is_a_readable_502(client, monkeypatch):
    monkeypatch.setattr(
        stt, "list_transcription_models",
        AsyncMock(side_effect=ExternalTranscriptionError("Couldn't load the list")),
    )
    response = client.get("/api/config/transcription_models")
    assert response.status_code == 502
    assert response.json()["detail"] == "Couldn't load the list"
