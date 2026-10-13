"""
Qdrant collection schema and initialization helpers.
"""
from __future__ import annotations

from typing import Any
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

logger = structlog.get_logger(__name__)


async def ensure_qdrant_collection(
    client: AsyncQdrantClient,
    collection_name: str = "document_chunks",
    vector_dimension: int = 384,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
    recreate: bool = False,
) -> bool:
    """
    Ensure the target Qdrant collection exists with proper vector configuration and payload indexes.
    """
    try:
        collections_response = await client.get_collections()
        existing_names = {c.name for c in collections_response.collections}

        if collection_name in existing_names and recreate:
            logger.info("recreating_qdrant_collection", collection=collection_name)
            await client.delete_collection(collection_name=collection_name)
            existing_names.remove(collection_name)

        if collection_name not in existing_names:
            logger.info(
                "creating_qdrant_collection",
                collection=collection_name,
                dimension=vector_dimension,
                distance=distance,
            )
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_dimension,
                    distance=distance,
                ),
            )

            # Create payload indexes for fast filtering
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="source_type",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            logger.info("qdrant_collection_created", collection=collection_name)

        return True
    except Exception as e:
        logger.error("qdrant_collection_creation_failed", collection=collection_name, error=str(e))
        raise
