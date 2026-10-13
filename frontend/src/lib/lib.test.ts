import { describe, expect, it } from "vitest";
import { classifyHttpError, isResearchAnswer } from "../api/client";
import { safeExternalUrl } from "./safeUrl";
import { buildSections, readPlan } from "./answerModel";
import type { ResearchAnswer } from "../types/api";

// NOTE: this fixture exists only for unit tests; it is never imported by the app.
function fixture(): ResearchAnswer {
  return {
    question: "q",
    summary: "s",
    sections: [
      {
        title: "Sec",
        claims: [
          { claim_id: "claim_001", text: "A is true.", evidence_ids: ["evidence_001"], citation_ids: [], support_status: "SUPPORTED" },
          { claim_id: "claim_009", text: "Unverified one.", evidence_ids: ["evidence_404"], citation_ids: [], support_status: "SUPPORTED" },
        ],
      },
    ],
    claims: [
      { claim_id: "claim_001", text: "A is true.", evidence_ids: ["evidence_001"], citation_ids: ["cite_1"], support_status: "PARTIALLY_SUPPORTED" },
      { claim_id: "claim_002", text: "Only top-level.", evidence_ids: [], citation_ids: [], support_status: "UNSUPPORTED" },
    ],
    citations: [
      {
        citation_id: "cite_1", number: 1, title: "T", source_type: "arxiv", source_id: "1",
        authors: [], publication_date: null, url: null, doi: null, arxiv_id: null,
        evidence_id: "evidence_001", formatted_citation: "[1] T",
      },
    ],
    sources: [],
    evidence: [],
    limitations: [],
    verification: {
      valid: true, errors: [], warnings: [], unsupported_claim_ids: [],
      verified_claim_count: 1, total_claim_count: 2, grounding_rate: 0.5,
    },
    metadata: {
      provider: "x", model: "y", planning_latency_ms: 0, retrieval_latency_ms: 0,
      generation_latency_ms: 0, verification_latency_ms: 0, total_latency_ms: 0,
      input_tokens: null, output_tokens: null, total_tokens: null, claim_count: 2,
      supported_claim_count: 0, partially_supported_claim_count: 1, unsupported_claim_count: 1,
      retrieval: null,
    },
    insufficient_evidence: false,
    query_analysis: null,
    research_plan: null,
  };
}

describe("safeExternalUrl", () => {
  it("allows only http(s)", () => {
    expect(safeExternalUrl("https://arxiv.org/abs/2005.11401")).toBe("https://arxiv.org/abs/2005.11401");
    expect(safeExternalUrl("javascript:alert(1)")).toBeNull();
    expect(safeExternalUrl("data:text/html,<script>")).toBeNull();
    expect(safeExternalUrl("not a url")).toBeNull();
    expect(safeExternalUrl(null)).toBeNull();
  });
});

describe("classifyHttpError", () => {
  it("maps index integrity errors without leaking detail", () => {
    const e = classifyHttpError(503, { error: "No active vector index exists.", code: "index_manifest_missing" });
    expect(e.kind).toBe("index_unavailable");
    expect(e.retryable).toBe(true);
  });
  it("maps embedding failures", () => {
    expect(classifyHttpError(503, { error: "x", code: "embedding_model_load_failed" }).kind).toBe("model_unavailable");
  });
  it("detects rate limits in 500 detail and hides the raw text", () => {
    const e = classifyHttpError(500, { detail: "Research answer synthesis failed: Gemini generation failed: 429 quota exceeded" });
    expect(e.kind).toBe("rate_limit");
    expect(e.message).not.toMatch(/Gemini generation failed/);
  });
  it("treats unknown 500s as generic server errors", () => {
    const e = classifyHttpError(500, { detail: "Traceback /home/user/secret.py" });
    expect(e.kind).toBe("server");
    expect(e.message).not.toMatch(/secret/);
  });
  it("maps the stable Phase 10 backend codes", () => {
    expect(classifyHttpError(429, { error: "slow down", code: "llm_rate_limited" }).kind).toBe("rate_limit");
    expect(classifyHttpError(504, { error: "t", code: "llm_timeout" }).kind).toBe("timeout");
    expect(classifyHttpError(502, { error: "m", code: "llm_model_unavailable" }).kind).toBe("model_unavailable");
    expect(classifyHttpError(400, { error: "u", code: "url_not_allowed" }).kind).toBe("invalid_request");
  });
  it("marks validation errors as non-retryable", () => {
    expect(classifyHttpError(422, {}).retryable).toBe(false);
  });
});

describe("isResearchAnswer", () => {
  it("accepts a valid shape and rejects malformed payloads", () => {
    expect(isResearchAnswer(fixture())).toBe(true);
    expect(isResearchAnswer({ summary: "x" })).toBe(false);
    expect(isResearchAnswer(null)).toBe(false);
  });
});

describe("buildSections", () => {
  it("takes status from verified top-level claims and never fabricates one", () => {
    const sections = buildSections(fixture());
    const [sec, extra] = sections;
    expect(sec.claims[0].status).toBe("PARTIALLY_SUPPORTED"); // from verified list, not section default
    expect(sec.claims[0].citations.map((c) => c.number)).toEqual([1]);
    expect(sec.claims[1].status).toBeNull(); // unmatched section claim => not verified
    expect(sec.claims[1].unknownEvidenceIds).toEqual(["evidence_404"]);
    expect(extra.title).toBe("Additional claims");
    expect(extra.claims[0].status).toBe("UNSUPPORTED");
  });
});

describe("readPlan", () => {
  it("returns null without plan data and parses subqueries defensively", () => {
    expect(readPlan(fixture())).toBeNull();
    const a = fixture();
    a.query_analysis = { intent: "COMPARATIVE", complexity: "COMPLEX", requires_decomposition: true };
    a.research_plan = { strategy: "parallel_independent", subqueries: [{ subquery_id: "sq_001", text: "x" }, 5] };
    const p = readPlan(a)!;
    expect(p.intent).toBe("COMPARATIVE");
    expect(p.subqueries).toHaveLength(1);
  });
});
