import { AlertTriangle, CircleHelp } from "lucide-react";
import type { Citation, ResearchAnswer } from "../types/api";
import { STATUS_LABEL, type ClaimView, type SectionView } from "../lib/answerModel";
import { Markdown } from "./Markdown";

interface Props {
  answer: ResearchAnswer;
  sections: SectionView[];
  selectedEvidenceId: string | null;
  highlightedClaimKey: string | null;
  onOpenCitation: (c: Citation) => void;
}

const STATUS_STYLE = {
  SUPPORTED: "bg-accent-tint text-accent",
  PARTIALLY_SUPPORTED: "bg-warn-tint text-warn",
  UNSUPPORTED: "bg-danger-tint text-danger",
} as const;

export function AnswerView({
  answer,
  sections,
  selectedEvidenceId,
  highlightedClaimKey,
  onOpenCitation,
}: Props) {
  const v = answer.verification;
  return (
    <article className="space-y-8" aria-label="Research answer">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-faint">Summary</h2>
        <div className="prose-answer font-serif text-[1.15rem] leading-[1.75] text-ink">
          <Markdown>{answer.summary}</Markdown>
        </div>
      </section>

      {sections.length > 0 && (
        <section aria-label="Claims">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-ink-faint">
            Findings &amp; supporting evidence
          </h2>
          <div className="space-y-6">
            {sections.map((sec, i) => (
              <div key={i}>
                {sec.title && <h3 className="mb-2 font-serif text-lg font-semibold">{sec.title}</h3>}
                <ul className="space-y-3">
                  {sec.claims.map((c) => (
                    <ClaimRow
                      key={c.key}
                      claim={c}
                      highlighted={highlightedClaimKey === c.key}
                      selectedEvidenceId={selectedEvidenceId}
                      onOpenCitation={onOpenCitation}
                    />
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      {answer.limitations.length > 0 && (
        <section aria-label="Limitations">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-faint">
            Limitations reported by the model
          </h2>
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-soft">
            {answer.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </section>
      )}

      {(v.errors.length > 0 || v.warnings.length > 0) && (
        <section
          className="rounded-lg border border-rule bg-warn-tint px-4 py-3 text-sm text-warn"
          aria-label="Citation check notes"
        >
          <p className="mb-1 flex items-center gap-2 font-medium">
            <AlertTriangle className="h-4 w-4" aria-hidden /> Citation check notes
          </p>
          <ul className="list-disc space-y-0.5 pl-5">
            {[...v.errors, ...v.warnings].map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}

function ClaimRow({
  claim,
  highlighted,
  selectedEvidenceId,
  onOpenCitation,
}: {
  claim: ClaimView;
  highlighted: boolean;
  selectedEvidenceId: string | null;
  onOpenCitation: (c: Citation) => void;
}) {
  return (
    <li
      id={`claim-${claim.key}`}
      className={`rounded-lg border px-4 py-3 transition-colors ${
        highlighted ? "border-accent bg-accent-tint" : "border-rule bg-paper-raised"
      }`}
    >
      <div className="font-serif text-[1.05rem] leading-relaxed">
        <Markdown inline>{claim.text}</Markdown>
        {claim.citations.map((c) => (
          <button
            key={c.citation_id}
            type="button"
            onClick={() => onOpenCitation(c)}
            aria-label={`Open citation ${c.number}: ${c.title}`}
            aria-pressed={selectedEvidenceId === c.evidence_id}
            className={`ml-1 inline-flex min-w-[1.6rem] items-center justify-center rounded px-1 align-baseline font-sans text-xs font-semibold ${
              selectedEvidenceId === c.evidence_id
                ? "bg-accent text-white"
                : "bg-accent-tint text-accent hover:bg-accent hover:text-white"
            }`}
          >
            {c.number}
          </button>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        {claim.status ? (
          <span className={`rounded-full px-2 py-0.5 font-medium ${STATUS_STYLE[claim.status]}`}>
            {STATUS_LABEL[claim.status]}
          </span>
        ) : (
          <span
            className="inline-flex items-center gap-1 rounded-full bg-paper-sunk px-2 py-0.5 font-medium text-ink-soft"
            title="The backend did not return a verification result for this claim."
          >
            <CircleHelp className="h-3 w-3" aria-hidden /> Not verified
          </span>
        )}
        {claim.citations.length === 0 && claim.unknownEvidenceIds.length === 0 && (
          <span className="text-ink-faint">No citation attached</span>
        )}
        {claim.unknownEvidenceIds.length > 0 && (
          <span className="text-danger">
            References evidence not present in the response ({claim.unknownEvidenceIds.join(", ")})
          </span>
        )}
      </div>
    </li>
  );
}
