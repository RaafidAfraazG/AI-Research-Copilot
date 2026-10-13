/**
 * View-model helpers: join claims, citations and evidence using only the ids the
 * backend returned. Nothing is invented; unknown references are surfaced as such.
 */
import type {
  AnswerEvidence,
  Citation,
  ClaimItem,
  PlanSubquery,
  ResearchAnswer,
  SupportStatus,
} from "../types/api";

export interface ClaimView {
  key: string;
  text: string;
  /** Present only when the claim was verified by the backend (top-level claims). */
  status: SupportStatus | null;
  citations: Citation[];
  evidenceIds: string[];
  /** Evidence ids the claim references that are missing from the response. */
  unknownEvidenceIds: string[];
}

export interface SectionView {
  title: string | null;
  claims: ClaimView[];
}

const norm = (s: string) => s.replace(/\s+/g, " ").trim().toLowerCase();

export function indexEvidence(answer: ResearchAnswer): Map<string, AnswerEvidence> {
  return new Map((answer.evidence ?? []).map((e) => [e.evidence_id, e]));
}

function toClaimView(
  claim: ClaimItem,
  key: string,
  verified: boolean,
  citById: Map<string, Citation>,
  citByEvidence: Map<string, Citation>,
  evidence: Map<string, AnswerEvidence>,
): ClaimView {
  const cits = new Map<string, Citation>();
  for (const cid of claim.citation_ids ?? []) {
    const c = citById.get(cid);
    if (c) cits.set(c.citation_id, c);
  }
  for (const eid of claim.evidence_ids ?? []) {
    const c = citByEvidence.get(eid);
    if (c) cits.set(c.citation_id, c);
  }
  const evidenceIds = claim.evidence_ids ?? [];
  const known = (id: string) => evidence.has(id) || citByEvidence.has(id);
  return {
    key,
    text: claim.text,
    status: verified ? claim.support_status : null,
    citations: [...cits.values()].sort((a, b) => a.number - b.number),
    evidenceIds,
    unknownEvidenceIds: evidenceIds.filter((id) => !known(id)),
  };
}

/**
 * Sections in the backend response carry claim objects that were not run through
 * the verifier (their support_status is just a default). Status is therefore taken
 * from the verified top-level `claims` list, matched by text; unmatched section
 * claims are shown without a status rather than with a fabricated one.
 */
export function buildSections(answer: ResearchAnswer): SectionView[] {
  const evidence = indexEvidence(answer);
  const citById = new Map(answer.citations.map((c) => [c.citation_id, c]));
  const citByEvidence = new Map(answer.citations.map((c) => [c.evidence_id, c]));
  const verifiedByText = new Map(answer.claims.map((c) => [norm(c.text), c]));
  const used = new Set<ClaimItem>();

  const out: SectionView[] = [];
  answer.sections.forEach((sec, si) => {
    const claims: ClaimView[] = [];
    sec.claims.forEach((sc, ci) => {
      const v = verifiedByText.get(norm(sc.text));
      if (v) used.add(v);
      claims.push(
        toClaimView(v ?? sc, `s${si}-c${ci}`, v !== undefined, citById, citByEvidence, evidence),
      );
    });
    if (claims.length) out.push({ title: sec.title, claims });
  });

  const rest = answer.claims.filter((c) => !used.has(c));
  if (rest.length) {
    out.push({
      title: out.length ? "Additional claims" : null,
      claims: rest.map((c, i) => toClaimView(c, `r${i}`, true, citById, citByEvidence, evidence)),
    });
  }
  return out;
}

/** Claims (any section) that cite a given evidence id, for back-navigation. */
export function claimsForEvidence(sections: SectionView[], evidenceId: string): ClaimView[] {
  return sections.flatMap((s) => s.claims).filter((c) => c.evidenceIds.includes(evidenceId));
}

export interface PlanView {
  intent: string | null;
  complexity: string | null;
  requiresDecomposition: boolean | null;
  strategy: string | null;
  subqueries: PlanSubquery[];
  entities: string[];
  concepts: string[];
}

const strArr = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];

/** Defensive read of the loosely typed plan/analysis dicts; null when absent. */
export function readPlan(answer: ResearchAnswer): PlanView | null {
  const qa = answer.query_analysis as Record<string, unknown> | null;
  const rp = answer.research_plan as Record<string, unknown> | null;
  if (!qa && !rp) return null;
  const subs = Array.isArray(rp?.subqueries) ? (rp?.subqueries as unknown[]) : [];
  return {
    intent: typeof qa?.intent === "string" ? qa.intent : null,
    complexity: typeof qa?.complexity === "string" ? qa.complexity : null,
    requiresDecomposition:
      typeof qa?.requires_decomposition === "boolean" ? qa.requires_decomposition : null,
    strategy: typeof rp?.strategy === "string" ? rp.strategy : null,
    subqueries: subs
      .filter((s): s is Record<string, unknown> => typeof s === "object" && s !== null)
      .map((s, i) => ({
        subquery_id: typeof s.subquery_id === "string" ? s.subquery_id : `sq_${i + 1}`,
        text: typeof s.text === "string" ? s.text : "",
        purpose: typeof s.purpose === "string" ? s.purpose : undefined,
        intent: typeof s.intent === "string" ? s.intent : undefined,
        priority: typeof s.priority === "number" ? s.priority : undefined,
        dependencies: strArr(s.dependencies),
      })),
    entities: strArr(qa?.entities),
    concepts: strArr(qa?.concepts),
  };
}

export const STATUS_LABEL: Record<SupportStatus, string> = {
  SUPPORTED: "Supported",
  PARTIALLY_SUPPORTED: "Partially supported",
  UNSUPPORTED: "Unsupported",
};

/** Human label for a source_type value; unknown types are shown as returned. */
export function sourceTypeLabel(t: string): string {
  const map: Record<string, string> = {
    arxiv: "arXiv",
    semantic_scholar: "Semantic Scholar",
    web: "Web",
    local: "Local corpus",
  };
  return map[t] ?? t;
}
