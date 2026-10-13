"""
Unit tests for ArxivConnector (connectors/arxiv_connector.py).
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest
import httpx

from connectors.arxiv_connector import ArxivConnector
from connectors.models import ConnectorSearchRequest

SAMPLE_ARXIV_ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">arXiv Query: search_query=all:RAG</title>
  <id>http://arxiv.org/api/12345</id>
  <updated>2026-09-29T00:00:00Z</updated>
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <updated>2021-04-12T18:00:00Z</updated>
    <published>2020-05-22T17:59:00Z</published>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>Large pre-trained language models store factual knowledge in their parameters.</summary>
    <author>
      <name>Patrick Lewis</name>
    </author>
    <author>
      <name>Ethan Perez</name>
    </author>
    <arxiv:doi>10.5555/3495724.3496517</arxiv:doi>
    <arxiv:comment>NeurIPS 2020</arxiv:comment>
    <link href="http://arxiv.org/abs/2005.11401v4" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2005.11401v4" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

EMPTY_ARXIV_ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">arXiv Query</title>
</feed>
"""


class TestArxivConnector:
    def test_extract_arxiv_id(self) -> None:
        connector = ArxivConnector()
        assert connector._extract_arxiv_id("http://arxiv.org/abs/2005.11401v4") == "2005.11401"
        assert connector._extract_arxiv_id("https://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
        assert connector._extract_arxiv_id("2005.11401") == "2005.11401"
        assert connector._extract_arxiv_id("cs/0112017") == "cs/0112017"

    def test_parse_atom_feed_valid_entry(self) -> None:
        connector = ArxivConnector()
        docs = connector._parse_atom_feed(SAMPLE_ARXIV_ATOM_XML)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "arxiv"
        assert doc.source_id == "2005.11401"
        assert doc.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
        assert doc.authors == ["Patrick Lewis", "Ethan Perez"]
        assert doc.publication_date == "2020-05-22"
        assert "cs.CL" in doc.categories
        assert doc.citation_info.arxiv_id == "2005.11401"
        assert doc.citation_info.doi == "10.5555/3495724.3496517"
        assert doc.pdf_url == "http://arxiv.org/pdf/2005.11401v4"

    def test_parse_empty_or_malformed_xml(self) -> None:
        connector = ArxivConnector()
        assert connector._parse_atom_feed("") == []
        assert connector._parse_atom_feed(EMPTY_ARXIV_ATOM_XML) == []
        assert connector._parse_atom_feed("<malformed><unclosed>") == []

    @pytest.mark.asyncio
    async def test_search_with_mocked_http(self) -> None:
        mock_http = AsyncMock()
        mock_http.get.return_value = httpx.Response(
            status_code=200,
            text=SAMPLE_ARXIV_ATOM_XML,
            request=httpx.Request("GET", "https://export.arxiv.org/api/query"),
        )

        connector = ArxivConnector(http_client=mock_http)
        res = await connector.search(ConnectorSearchRequest(query="RAG", limit=5))

        assert res.source_type == "arxiv"
        assert res.total_results == 1
        assert len(res.documents) == 1
        assert res.documents[0].source_id == "2005.11401"
        assert mock_http.get.called

    @pytest.mark.asyncio
    async def test_fetch_with_mocked_http(self) -> None:
        mock_http = AsyncMock()
        mock_http.get.return_value = httpx.Response(
            status_code=200,
            text=SAMPLE_ARXIV_ATOM_XML,
            request=httpx.Request("GET", "https://export.arxiv.org/api/query"),
        )

        connector = ArxivConnector(http_client=mock_http)
        doc = await connector.fetch("2005.11401")
        assert doc is not None
        assert doc.source_id == "2005.11401"
