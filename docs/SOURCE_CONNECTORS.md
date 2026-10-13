# Multi-Source Research Ingestion & Connectors (Phase 3)

## Overview

Phase 3 introduces a robust, multi-source ingestion layer that enables **AI Research Copilot** to search, fetch, normalize, deduplicate, and index external academic and web documents. The retrieval pipeline remains completely source-agnostic while preserving full citation provenance.

---

## 1. Architecture

```
                  ┌─────────────────┐
                  │ External Sources│
                  │ (arXiv, S2, Web)│
                  └────────┬────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │  ResearchSourceConnector (ABC)   │
          │  ├─ ArxivConnector               │
          │  ├─ SemanticScholarConnector     │
          │  └─ WebFetcher                   │
          └────────────────┬─────────────────┘
                           │ Typed ResearchDocument
                           ▼
          ┌──────────────────────────────────┐
          │ DocumentDeduplicator & Normalizer│
          │  ├─ Canonical ID resolution      │
          │  ├─ URL param stripping          │
          │  ├─ Content hashing              │
          │  └─ Multi-source metadata merge  │
          └────────────────┬─────────────────┘
                           │ Deduplicated ResearchDocument
                           ▼
          ┌──────────────────────────────────┐
          │    ConnectorIngestionService     │
          └────────────────┬─────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │    IndexingService    │
               │  ├─ Passage Chunking  │
               │  ├─ Dense Embedding   │
               │  ├─ BM25 Index        │
               │  ├─ Vector Index      │
               │  └─ PostgreSQL Meta   │
               └───────────────────────┘
```

---

## 2. Normalized Research Document Model

All connectors produce a canonical internal representation defined in [`backend/connectors/models.py`](../backend/connectors/models.py):

* **`ResearchDocument`**:
  * `id`: Internal UUID
  * `source_type`: `arxiv` | `semantic_scholar` | `web` | `local`
  * `source_id`: Native source identifier (e.g. arXiv ID `2005.11401` or URL)
  * `title`: Canonical title
  * `authors`: List of authors
  * `abstract`: Abstract or executive summary
  * `full_text`: Full body text when available
  * `url`: Canonical web link
  * `pdf_url`: Direct PDF link when available
  * `publication_date`: ISO format publication date
  * `venue`: Journal, conference, or domain
  * `categories`: Subject taxonomy / tags (e.g. `cs.CL`, `cs.AI`)
  * `citation_info`: Structured citation metadata (`CitationInfo`)
  * `provenance`: Detailed provenance tracing (`ProvenanceInfo`)
  * `metadata`: Raw & extended metadata

* **`CitationInfo`**:
  * `citation_count`: Global citation count
  * `reference_count`: Number of cited references
  * `influential_citation_count`: High-impact citations
  * `tldr`: High-level summary from model or publisher
  * `doi`: Digital Object Identifier
  * `arxiv_id`: Clean arXiv identifier
  * `corpus_id`: External corpus reference

* **`ProvenanceInfo`**:
  * `source_type`: Original ingestion source
  * `source_id`: Source identifier
  * `ingested_at`: UTC timestamp of ingestion
  * `canonical_url`: Normalized URL
  * `cross_source_ids`: Mappings across sources (`arxiv`, `doi`, `semantic_scholar`, `url`)

---

## 3. Connectors

### A. arXiv Connector ([`backend/connectors/arxiv_connector.py`](../backend/connectors/arxiv_connector.py))
* **Endpoint**: Official arXiv Atom XML API (`https://export.arxiv.org/api/query`)
* **Capabilities**: Keyword search, pagination, paper lookup by arXiv ID (e.g., `2005.11401` or `cs/0112017`).
* **Metadata Extracted**: Title, authors, abstract, publication date, primary/secondary categories, DOI, direct PDF links.
* **Authentication**: No API key required.

### B. Semantic Scholar Connector ([`backend/connectors/semantic_scholar_connector.py`](../backend/connectors/semantic_scholar_connector.py))
* **Endpoint**: Academic Graph REST API (`https://api.semanticscholar.org/graph/v1`)
* **Capabilities**: Search by keyword or year, fetch by Semantic Scholar Paper ID, DOI, or arXiv ID.
* **Metadata Extracted**: Citation metrics (citation count, influential citations), TLDR summaries, venue, external IDs (ArXiv, DOI, CorpusId).
* **Authentication**: Optional `SEMANTIC_SCHOLAR_API_KEY`. Operates reliably with automatic rate pacing and retry backoff if no key is provided.

### C. Web Fetcher ([`backend/connectors/web_fetcher.py`](../backend/connectors/web_fetcher.py))
* **Capabilities**: Single-URL safe fetching, boilerplate stripping, title and metadata extraction (`og:title`, `meta description`, `canonical` link).
* **HTML Parsing**: Robust standard library parser (`HTMLContentExtractor`) with void tag awareness and fallback to `BeautifulSoup4` if installed.

---

## 4. Security Controls & SSRF Protection

The `WebFetcher` incorporates strict SSRF defenses to prevent arbitrary network probing or access to internal infrastructure:

1. **Scheme Validation**: Enforces `http` or `https` only (rejects `file://`, `ftp://`, `gopher://`).
2. **DNS Resolution & IP Inspection**: Resolves hostnames via `socket.getaddrinfo` and checks against:
   * Loopback addresses (`127.0.0.0/8`, `::1`)
   * Private RFC 1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
   * Link-local addresses (`169.254.0.0/16`, `fe80::/10`)
   * Cloud metadata services (`169.254.169.254`)
   * Multicast and reserved ranges
3. **Payload Size Limits**: Enforces a configurable `max_bytes` ceiling (default: 5MB) before parsing to prevent memory exhaustion attacks.
4. **Content-Type Validation**: Rejects binary files, executables, audio, video, and archives.

---

## 5. Rate Limiting & Resilient HTTP Client

External HTTP calls are managed centrally by [`backend/connectors/http_client.py`](../backend/connectors/http_client.py) (`SafeAsyncHttpClient`):

* **Rate Pacing**: Configurable minimum delay interval between requests to external hosts to prevent hammering APIs.
* **Exponential Backoff**: Jittered exponential backoff for retryable errors.
* **Retryable Status Codes**: Automatically retries HTTP `429`, `500`, `502`, `503`, and `504` responses.
* **Header & Secret Sanitization**: Masks `x-api-key`, `Authorization`, and `Bearer` headers in logs.

---

## 6. Deduplication & Cross-Source Identity Resolution

Document deduplication is implemented in [`backend/connectors/deduplication.py`](../backend/connectors/deduplication.py):

* **Canonical ID Hierarchy**:
  1. `arxiv:<id>` (version stripped, e.g. `arxiv:2005.11401`)
  2. `doi:<doi>` (lowercased DOI)
  3. `url:<normalized_url>` (stripped of tracking query params `utm_*`, `ref`, `fbclid`, etc.)
  4. `<source_type>:<source_id>`
* **Metadata Merging**: When a document is ingested from multiple sources (e.g. arXiv first, then Semantic Scholar):
  * Merges authors and categories without duplicates
  * Enriches document with citation counts and TLDR summaries from Semantic Scholar
  * Preserves all source IDs in `provenance.cross_source_ids`
  * Avoids duplicate chunk indexing

---

## 7. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/sources/arxiv/search?query=...&limit=10` | Search arXiv papers |
| `GET` | `/api/v1/sources/semantic-scholar/search?query=...&limit=10` | Search Semantic Scholar |
| `GET` | `/api/v1/sources/web/fetch?url=...` | Safely fetch & extract web page |
| `POST` | `/api/v1/sources/ingest` | Fetch and index papers into retrieval pipeline |

---

## 8. CLI Verification Scripts

```powershell
# 1. Search arXiv
python scripts/test_arxiv.py "retrieval augmented generation" --limit 3

# 2. Search Semantic Scholar
python scripts/test_semantic_scholar.py "dense passage retrieval" --limit 3

# 3. Safely fetch web document
python scripts/test_web_fetcher.py "https://en.wikipedia.org/wiki/Information_retrieval"

# 4. Ingest and index external document directly into retrieval pipeline
python scripts/ingest_source.py arxiv "cross-encoder reranking" --limit 1

# 5. Verify retrieval of ingested document
python scripts/verify_retrieval.py "medical procedure reranking"
```
