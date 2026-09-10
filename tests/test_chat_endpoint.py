import httpx2
import pytest
import anthropic
from unittest.mock import MagicMock
from fastapi import HTTPException

from src.utils.config import settings
import src.api.routes as routes
from src.api.schemas import ChatRequest, ChatMessage


def _fake_response(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _api_error(error_cls, status_code, message="error"):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx2.Response(status_code, request=req, headers={})
    return error_cls(message, response=resp, body=None)


def test_chat_returns_503_when_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    with pytest.raises(HTTPException) as exc_info:
        routes.chat(ChatRequest(message="Why is wind low?"))
    assert exc_info.value.status_code == 503


def test_chat_calls_claude_with_context_and_history(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "fake-key")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response("Wind is low due to a calm weather system.")
    monkeypatch.setattr(routes.anthropic, "Anthropic", lambda api_key: fake_client)

    request = ChatRequest(
        message="Why is wind onshore low tomorrow?",
        history=[ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")],
        context={"forecast": [{"timestamp": "2026-01-01T00:00:00", "wind_onshore_mw": 100.0}]},
    )
    result = routes.chat(request)

    assert result.reply == "Wind is low due to a calm weather system."
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == settings.CHAT_MODEL
    assert len(call_kwargs["messages"]) == 3  # 2 history entries + the new question
    assert call_kwargs["messages"][-1] == {"role": "user", "content": "Why is wind onshore low tomorrow?"}
    assert "wind_onshore_mw" in call_kwargs["system"][0]["text"]


def test_chat_maps_authentication_error_to_503(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "bad-key")
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = _api_error(anthropic.AuthenticationError, 401)
    monkeypatch.setattr(routes.anthropic, "Anthropic", lambda api_key: fake_client)

    with pytest.raises(HTTPException) as exc_info:
        routes.chat(ChatRequest(message="test"))
    assert exc_info.value.status_code == 503


def test_chat_maps_rate_limit_error_to_429(monkeypatch):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "fake-key")
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = _api_error(anthropic.RateLimitError, 429)
    monkeypatch.setattr(routes.anthropic, "Anthropic", lambda api_key: fake_client)

    with pytest.raises(HTTPException) as exc_info:
        routes.chat(ChatRequest(message="test"))
    assert exc_info.value.status_code == 429
