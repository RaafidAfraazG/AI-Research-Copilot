"""
Unit tests for SafeAsyncHttpClient (connectors/http_client.py).
"""
from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from connectors.http_client import SafeAsyncHttpClient


class TestSafeAsyncHttpClient:
    def test_sanitize_headers_masks_secrets(self) -> None:
        client = SafeAsyncHttpClient()
        headers = {
            "User-Agent": "AI-Research-Copilot",
            "x-api-key": "secret_key_12345",
            "Authorization": "Bearer token_abcde",
            "Content-Type": "application/json",
        }
        sanitized = client._sanitize_headers(headers)
        assert sanitized["User-Agent"] == "AI-Research-Copilot"
        assert sanitized["x-api-key"] == "[REDACTED]"
        assert sanitized["Authorization"] == "[REDACTED]"
        assert sanitized["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_get_successful_response(self) -> None:
        client = SafeAsyncHttpClient(timeout_seconds=5.0, min_request_interval_seconds=0.0)
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = httpx.Response(status_code=200, json={"status": "ok"})
            mock_get.return_value = mock_resp

            res = await client.get("https://api.example.com/test")
            assert res.status_code == 200
            assert res.json() == {"status": "ok"}
            assert mock_get.call_count == 1
