"""
Grounded research synthesis and citation engine module.
"""
from synthesis.answer_service import (
    ResearchAnswerRequest,
    ResearchAnswerService,
)
from synthesis.citation_mapper import CitationMapper
from synthesis.citation_verifier import CitationVerifier
from synthesis.models import (
    Citation,
    CitationVerificationResult,
    ClaimItem,
    ResearchAnswer,
    ResearchAnswerSection,
    SourceProvenance,
    SupportStatus,
    SynthesisMetadata,
)
from synthesis.prompt_builder import GroundedPromptBuilder

__all__ = [
    "SupportStatus",
    "ClaimItem",
    "Citation",
    "SourceProvenance",
    "CitationVerificationResult",
    "ResearchAnswerSection",
    "SynthesisMetadata",
    "ResearchAnswer",
    "ResearchAnswerRequest",
    "ResearchAnswerService",
    "GroundedPromptBuilder",
    "CitationMapper",
    "CitationVerifier",
]
