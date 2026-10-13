"""
API v1 router — aggregates all v1 endpoint routers.

New routers (research, sources, etc.) added in future phases should be
included here to keep main.py clean.
"""
from fastapi import APIRouter

from api.v1.health import router as health_router
from api.v1.research import router as research_router
from api.v1.retrieval import router as retrieval_router
from api.v1.sources import router as sources_router

api_v1_router = APIRouter()

# ── Registered routers ────────────────────────────────────────────────────────
api_v1_router.include_router(health_router)
api_v1_router.include_router(retrieval_router)
api_v1_router.include_router(sources_router)
api_v1_router.include_router(research_router)



