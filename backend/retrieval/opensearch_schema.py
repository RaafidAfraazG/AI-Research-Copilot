"""
OpenSearch index schema, mappings, and index initialization helpers.
"""
from __future__ import annotations

from typing import Any
import structlog
from opensearchpy import AsyncOpenSearch

logger = structlog.get_logger(__name__)

OPENSEARCH_DOCUMENTS_INDEX_MAPPINGS: dict[str, Any] = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "research_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "stop",
                            "english_stemmer",
                        ],
                    }
                },
                "filter": {
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english",
                    }
                },
            },
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "source_type": {"type": "keyword"},
            "source_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "research_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 512}
                },
            },
            "url": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "research_analyzer",
            },
            "chunk_index": {"type": "integer"},
            "total_chunks": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "metadata": {"type": "object", "dynamic": True},
            "indexed_at": {"type": "date"},
        }
    },
}


async def ensure_opensearch_index(
    client: AsyncOpenSearch,
    index_name: str = "documents",
    recreate: bool = False,
) -> bool:
    """
    Ensure the target OpenSearch index exists with the required schema.
    """
    try:
        exists = await client.indices.exists(index=index_name)
        if exists and recreate:
            logger.info("recreating_opensearch_index", index=index_name)
            await client.indices.delete(index=index_name)
            exists = False

        if not exists:
            logger.info("creating_opensearch_index", index=index_name)
            await client.indices.create(
                index=index_name,
                body=OPENSEARCH_DOCUMENTS_INDEX_MAPPINGS,
            )
            logger.info("opensearch_index_created", index=index_name)
        return True
    except Exception as e:
        logger.error("opensearch_index_creation_failed", index=index_name, error=str(e))
        raise
