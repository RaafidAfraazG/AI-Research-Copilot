"""
Unit tests for SemanticScholarConnector (connectors/semantic_scholar_connector.py).
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest
import httpx

from connectors.models import ConnectorSearchRequest
from connectors.semantic_scholar_connector import SemanticScholarConnector

SAMPLE_S2_SEARCH_JSON = {
    "total": 1,
    "offset": 0,
    "data": [
        {
            "paperId": "a1b2c3d4e5f6",
            "title": "Dense Passage Retrieval for Open-Domain Question Answering",
            "abstract": "Open-domain question answering relies on efficient passage retrieval.",
            "url": "https://www.semanticscholar.org/paper/a1b2c3d4e5f6",
            "venue": "EMNLP 2020",
            "year": 2020,
            "citationCount": 2400,
            "authors": [
                {"authorId": "1", "name": "Vladimir Karpukhin"},
                {"authorId": "2", "name": "Barlas Oguz"},
            ],
            "fieldsOfStudy": ["Computer Science"],
            "tldr": {
                "model": "tldr@v2.0.0",
                "text": "This paper shows that retrieval can be implemented using dense representations."
            },
            "externalIds": {
                "ArXiv": "2004.04906",
                "DOI": "10.18653/v1/2020.emnlp-main.550"
            }
        }
    ]
}


class TestSemanticScholarConnector:
    def test_normalize_paper(self) -> None:
        connector = SemanticScholarConnector()
        doc = connector._parse_paper_dict(SAMPLE_S2_SEARCH_JSON["data"][0])
        assert doc is not None
        assert doc.source_type == "semantic_scholar"
        assert doc.source_id == "a1b2c3d4e5f6"
        assert "Dense Passage Retrieval" in doc.title
        assert doc.authors == ["Vladimir Karpukhin", "Barlas Oguz"]
        assert doc.citation_info.citation_count == 2400
        assert doc.citation_info.arxiv_id == "2004.04906"
        assert doc.citation_info.doi == "10.18653/v1/2020.emnlp-main.550"
        assert "dense representations" in (doc.citation_info.tldr or "")
        assert doc.provenance.cross_source_ids["arxiv"] == "2004.04906"

    def test_normalize_paper_missing_fields(self) -> None:
        connector = SemanticScholarConnector()
        minimal_item = {"paperId": "min123", "title": "Minimal Paper"}
        doc = connector._parse_paper_dict(minimal_item)
        assert doc is not None
        assert doc.source_id == "min123"
        assert doc.title == "Minimal Paper"
        assert doc.authors == []
        assert doc.citation_info.citation_count is None

    @pytest.mark.asyncio
    async def test_search_with_mocked_http(self) -> None:
        mock_http = AsyncMock()
        mock_http.get.return_value = httpx.Response(
            status_code=200,
            json=SAMPLE_S2_SEARCH_JSON,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
        )

        connector = SemanticScholarConnector(http_client=mock_http)
        res = await connector.search(ConnectorSearchRequest(query="DPR", limit=5))

        assert res.source_type == "semantic_scholar"
        assert res.total_results == 1
        assert len(res.documents) == 1
        assert res.documents[0].source_id == "a1b2c3d4e5f6"

    @pytest.mark.asyncio
    async def test_fetch_with_mocked_http(self) -> None:
        mock_http = AsyncMock()
        mock_http.get.return_value = httpx.Response(
            status_code=200,
            json=SAMPLE_S2_SEARCH_JSON["data"][0],
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/a1b2c3d4e5f6"),
        )

        connector = SemanticScholarConnector(http_client=mock_http)
        doc = await connector.fetch("a1b2c3d4e5f6")
        assert doc is not None
        assert doc.source_id == "a1b2c3d4e5f6"
        assert doc.citation_info.citation_count == 2400
