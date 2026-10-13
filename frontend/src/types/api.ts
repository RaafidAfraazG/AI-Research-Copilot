/**
 * Types mirroring the FastAPI schemas (backend/synthesis/models.py and
 * backend/api/v1/*). Keep in sync with the backend; do not add fields the
 * backend does not return.
 */

export type SupportStatus = "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED";

export interface ClaimItem {
  claim_id: string;
  text: string;
  evidence_ids: string[];
  citation_ids: string[];
  support_status: SupportStatus;
}

export interface Citation {
  citation_id: string;
  number: number;
  title: string;
  source_type: string;
  source_id: string;
  authors: string[];
  publication_date: string | null;
  url: string | null;
  doi: string | null;
  arxiv_id: string | null;
  evidence_id: string;
  formatted_citation: string;
}

export interface SourceProvenance {
  source_id: string;
  source_type: string;
  title: string;
  url: string | null;
  evidence_count: number;
}

export interface AnswerEvidence {
  evidence_id: string;
  passage_id: string;
  document_id: string;
  title: string;
  source_type: string;
  source_id: string;
  url: string | null;
  text: string;
  relevance_score: number;
  rank: number;
  chunk_index: number;
  authors: string[];
  publication_date: string | null;
}

export interface CitationVerificationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  unsupported_claim_ids: string[];
  verified_claim_count: number;
  total_claim_count: number;
  grounding_rate: number;
  /** Phase 10: how support was determined (structural evidence-id check). */
  verification_method?: string;
  /** Phase 10: false => no entailment check was performed. */
  semantic_entailment_checked?: boolean;
}

export interface AnswerSection {
  title: string;
  claims: ClaimItem[];
}

export interface SynthesisMetadata {
  provider: string;
  model: string;
  planning_latency_ms: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  verification_latency_ms: number;
  total_latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  claim_count: number;
  supported_claim_count: number;
  partially_supported_claim_count: number;
  unsupported_claim_count: number;
  retrieval: Record<string, unknown> | null;
}

export interface ResearchAnswer {
  question: string;
  summary: string;
  sections: AnswerSection[];
  claims: ClaimItem[];
  citations: Citation[];
  sources: SourceProvenance[];
  /** Added in Phase 9; absent on older backends. */
  evidence?: AnswerEvidence[];
  limitations: string[];
  verification: CitationVerificationResult;
  metadata: SynthesisMetadata;
  insufficient_evidence: boolean;
  query_analysis: Record<string, unknown> | null;
  research_plan: Record<string, unknown> | null;
}

export interface ResearchAnswerRequest {
  query: string;
  enable_decomposition?: boolean;
  decomposition_mode?: "auto" | "llm" | "deterministic";
  max_subqueries?: number;
  candidate_k?: number;
  evidence_k?: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  timestamp: string;
}

/** GET /retrieval/index/status */
export interface IndexStatus {
  index_present: boolean;
  compatible: boolean;
  reason?: string;
  detail?: string;
  generation_id?: string;
  status?: string;
  created_at?: string;
  embedding?: {
    provider: string;
    model: string;
    dimension: number;
    revision: string | null;
    normalized: boolean;
    is_mock: boolean;
  };
  counts?: { documents: number; chunks: number };
  embedding_provider_error?: string;
}

/** Shape of one subquery nested in ResearchAnswer.research_plan. */
export interface PlanSubquery {
  subquery_id: string;
  text: string;
  purpose?: string;
  intent?: string;
  priority?: number;
  dependencies?: string[];
}
