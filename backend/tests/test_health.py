"""
Tests for health and readiness endpoints.

GET /api/v1/health  — liveness probe
GET /api/v1/ready   — readiness probe (infrastructure checks)

All infrastructure checks are mocked via the `mock_infra` fixture
so tests don't require running Docker services.
"""
from __future__ import annotations

from httpx import AsyncClient


class TestHealthEndpoint:
    """Tests for GET /api/v1/health"""

    async def test_returns_200(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/health")
        assert response.status_code == 200

    async def test_response_schema(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data

    async def test_returns_timing_header(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/health")
        assert "x-process-time-ms" in response.headers

    async def test_content_type_is_json(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/api/v1/health")
        assert "application/json" in response.headers["content-type"]


class TestReadyEndpoint:
    """Tests for GET /api/v1/ready"""

    async def test_returns_200_when_all_services_up(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        response = await api_client.get("/api/v1/ready")
        assert response.status_code == 200

    async def test_response_schema_when_ready(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        response = await api_client.get("/api/v1/ready")
        data = response.json()
        assert data["status"] == "ready"
        assert "checks" in data
        assert "timestamp" in data

    async def test_all_checks_reported(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        response = await api_client.get("/api/v1/ready")
        checks = response.json()["checks"]
        for name in ("postgres", "index", "llm", "opensearch", "qdrant", "redis"):
            assert name in checks

    async def test_optional_services_are_skipped_and_not_required(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        """OpenSearch/Qdrant-server/Redis are not part of the local architecture."""
        checks = (await api_client.get("/api/v1/ready")).json()["checks"]
        for name in ("opensearch", "qdrant", "redis"):
            assert checks[name]["status"] == "skipped"
            assert checks[name]["required"] is False
        # skipped probes are never executed
        assert not mock_infra["opensearch"].called
        assert not mock_infra["redis"].called

    async def test_returns_503_when_postgres_down(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        from api.v1.health import CheckResult
        mock_infra["postgres"].return_value = CheckResult(
            status="error", error="Connection refused"
        )
        response = await api_client.get("/api/v1/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["postgres"]["status"] == "error"

    async def test_returns_503_when_index_unusable(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        from api.v1.health import CheckResult
        mock_infra["index"].return_value = CheckResult(status="error", error="index_manifest_missing")
        response = await api_client.get("/api/v1/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["index"]["error"] == "index_manifest_missing"

    async def test_503_response_includes_ok_services(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        """When one required check fails, the others should still report ok."""
        from api.v1.health import CheckResult
        mock_infra["llm"].return_value = CheckResult(
            status="error", error="gemini_api_key_not_configured"
        )
        response = await api_client.get("/api/v1/ready")
        data = response.json()
        assert data["checks"]["postgres"]["status"] == "ok"
        assert data["checks"]["index"]["status"] == "ok"
        assert data["checks"]["llm"]["status"] == "error"

    async def test_ready_has_timing_header(
        self, api_client: AsyncClient, mock_infra
    ) -> None:
        response = await api_client.get("/api/v1/ready")
        assert "x-process-time-ms" in response.headers


class TestSwaggerDocs:
    """Ensure the auto-generated docs are accessible."""

    async def test_openapi_schema_accessible(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "AI Research Copilot API"

    async def test_swagger_ui_accessible(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/docs")
        assert response.status_code == 200
