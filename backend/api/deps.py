"""
Process-level service registry for the API layer.

Services own network clients (OpenSearch/Qdrant clients) and, through the shared
reranker, a loaded neural model. Creating them per request leaked client sessions and
reloaded the cross-encoder on every call, so the API shares one instance of each and
closes them on application shutdown (see main.lifespan).
"""
from __future__ import annotations

import threading
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_lock = threading.RLock()  # factories re-enter (answer -> planning -> retrieval)
_instances: dict[str, Any] = {}


def _get(name: str, factory: Any) -> Any:
    with _lock:
        inst = _instances.get(name)
        if inst is None:
            inst = factory()
            _instances[name] = inst
        return inst


def get_retrieval_service():  # -> ResearchRetrievalService
    from services.research_retrieval_service import ResearchRetrievalService

    return _get("retrieval", ResearchRetrievalService)


def get_planning_service():  # -> ResearchPlanningService
    from query.planner import ResearchPlanningService

    return _get("planning", lambda: ResearchPlanningService(retrieval_service=get_retrieval_service()))


def get_answer_service():  # -> ResearchAnswerService
    from synthesis.answer_service import ResearchAnswerService

    return _get(
        "answer",
        lambda: ResearchAnswerService(
            retrieval_service=get_retrieval_service(), planning_service=get_planning_service()
        ),
    )


def get_hybrid_retriever():  # -> HybridRetriever (shared with the retrieval service)
    return get_retrieval_service().hybrid_retriever


def get_indexing_service():  # -> IndexingService
    from retrieval.indexing_service import IndexingService

    return _get("indexing", IndexingService)


def reset_services() -> None:
    """Forget cached instances without closing them (tests / settings changes)."""
    with _lock:
        _instances.clear()


async def aclose_services() -> None:
    """Close network clients held by shared services. Safe to call more than once."""
    with _lock:
        items = list(_instances.items())
        _instances.clear()

    async def _close(obj: Any) -> None:
        closer = getattr(obj, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception as e:  # shutdown must not raise
                log.warning("service_close_failed", error_type=type(e).__name__)

    for _, inst in items:
        await _close(inst)
        # services that own a retriever / indexer
        for attr in ("hybrid_retriever", "retrieval_service"):
            sub = getattr(inst, attr, None)
            if sub is not None and sub is not inst:
                await _close(sub)
