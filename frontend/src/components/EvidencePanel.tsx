import { useEffect, useRef } from "react";
import { ExternalLink, FileWarning, X } from "lucide-react";
import type { AnswerEvidence, Citation, ResearchAnswer } from "../types/api";
import {
  claimsForEvidence,
  sourceTypeLabel,
  STATUS_LABEL,
  type SectionView,
} from "../lib/answerModel";
import { safeExternalUrl } from "../lib/safeUrl";

interface Props {
  answer: ResearchAnswer;
  sections: SectionView[];
  selectedEvidenceId: string | null;
  onSelect: (evidenceId: string | null) => void;
  onClose: () => void;
  onJumpToClaim: (claimKey: string) => void;
  overlay: boolean;
}

interface Entry {
  evidenceId: string;
  evidence: AnswerEvidence | null;
  citation: Citation | null;
}

export function EvidencePanel({
  answer,
  sections,
  selectedEvidenceId,
  onSelect,
  onClose,
  onJumpToClaim,
  overlay,
}: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const selectedRef = useRef<HTMLLIElement>(null);

  const citByEvidence = new Map(answer.citations.map((c) => [c.evidence_id, c]));
  const entries: Entry[] = (answer.evidence ?? []).map((e) => ({
    evidenceId: e.evidence_id,
    evidence: e,
    citation: citByEvidence.get(e.evidence_id) ?? null,
  }));
  // Citations whose passage text is not in the response are shown as unavailable.
  const have = new Set(entries.map((e) => e.evidenceId));
  for (const c of answer.citations) {
    if (!have.has(c.evidence_id)) {
      entries.push({ evidenceId: c.evidence_id, evidence: null, citation: c });
    }
  }
  // Selected id that is entirely unknown to the response.
  if (selectedEvidenceId && !entries.some((e) => e.evidenceId === selectedEvidenceId)) {
    entries.unshift({ evidenceId: selectedEvidenceId, evidence: null, citation: null });
  }

  useEffect(() => {
    if (overlay) closeRef.current?.focus();
  }, [overlay]);

  useEffect(() => {
    if (!overlay) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [overlay, onClose]);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedEvidenceId]);

  return (
    <aside
      aria-label="Evidence and sources"
      role={overlay ? "dialog" : "complementary"}
      aria-modal={overlay ? true : undefined}
      className="flex h-full min-h-0 flex-col bg-paper-raised"
    >
      <header className="flex items-center gap-2 border-b border-rule px-4 py-3">
        <h2 className="text-sm font-semibold">
          Evidence <span className="font-normal text-ink-faint">({entries.length})</span>
        </h2>
        <button
          ref={closeRef}
          type="button"
          onClick={onClose}
          aria-label="Close evidence panel"
          className="ml-auto rounded-md p-1.5 text-ink-soft hover:bg-paper-sunk"
        >
          <X className="h-4 w-4" aria-hidden />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {entries.length === 0 ? (
          <p className="px-4 py-6 text-sm text-ink-soft">
            No evidence passages were returned for this question.
          </p>
        ) : (
          <ul className="divide-y divide-rule">
            {entries.map((en) => {
              const selected = en.evidenceId === selectedEvidenceId;
              return (
                <li key={en.evidenceId} ref={selected ? selectedRef : undefined}>
                  <EvidenceItem
                    entry={en}
                    selected={selected}
                    claims={claimsForEvidence(sections, en.evidenceId)}
                    onToggle={() => onSelect(selected ? null : en.evidenceId)}
                    onJumpToClaim={onJumpToClaim}
                  />
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <footer className="border-t border-rule px-4 py-2.5 text-xs leading-relaxed text-ink-faint">
        Support status shows whether a claim&rsquo;s cited evidence exists in the retrieved set. It
        does not prove the passage entails the claim &mdash; read the passage. Scores are ranking
        scores, not probabilities.
      </footer>
    </aside>
  );
}

function EvidenceItem({
  entry,
  selected,
  claims,
  onToggle,
  onJumpToClaim,
}: {
  entry: Entry;
  selected: boolean;
  claims: ReturnType<typeof claimsForEvidence>;
  onToggle: () => void;
  onJumpToClaim: (claimKey: string) => void;
}) {
  const { evidence: ev, citation: cit } = entry;
  const title = ev?.title || cit?.title || "Unknown source";
  const url = safeExternalUrl(ev?.url ?? cit?.url);
  const authors = ev?.authors?.length ? ev.authors : (cit?.authors ?? []);
  const date = ev?.publication_date ?? cit?.publication_date ?? null;
  const sourceType = ev?.source_type ?? cit?.source_type;

  return (
    <div className={selected ? "bg-accent-tint/50" : undefined}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={selected}
        className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-paper-sunk/60"
      >
        <span
          className={`mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded px-1 text-xs font-semibold ${
            cit ? "bg-accent text-white" : "bg-paper-sunk text-ink-soft"
          }`}
          title={cit ? `Citation ${cit.number}` : "Retrieved but not cited"}
        >
          {cit ? cit.number : "–"}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium leading-snug">{title}</span>
          <span className="mt-0.5 block text-xs text-ink-faint">
            {sourceType ? sourceTypeLabel(sourceType) : "Source"}
            {ev ? ` · rank ${ev.rank}` : ""}
            {!cit && ev ? " · not cited by any claim" : ""}
          </span>
        </span>
      </button>

      {selected && (
        <div className="space-y-3 px-4 pb-4 text-sm">
          {ev ? (
            <blockquote className="whitespace-pre-wrap rounded-lg border-l-2 border-accent bg-paper px-3 py-2 font-serif text-[0.95rem] leading-relaxed">
              {ev.text}
            </blockquote>
          ) : (
            <p className="flex items-start gap-2 rounded-lg bg-warn-tint px-3 py-2 text-warn">
              <FileWarning className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              The passage text for this citation is not available in the response.
            </p>
          )}

          <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-xs">
            {authors.length > 0 && (
              <>
                <dt className="text-ink-faint">Authors</dt>
                <dd>{authors.join(", ")}</dd>
              </>
            )}
            {date && (
              <>
                <dt className="text-ink-faint">Published</dt>
                <dd>{date}</dd>
              </>
            )}
            {(ev?.source_id ?? cit?.source_id) && (
              <>
                <dt className="text-ink-faint">Source ID</dt>
                <dd className="break-all">{ev?.source_id ?? cit?.source_id}</dd>
              </>
            )}
            {ev && (
              <>
                <dt className="text-ink-faint">Ranking score</dt>
                <dd>
                  {ev.relevance_score.toFixed(3)}{" "}
                  <span className="text-ink-faint">(passage {ev.chunk_index + 1})</span>
                </dd>
              </>
            )}
          </dl>

          {url && (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-accent underline underline-offset-2"
            >
              Open source <ExternalLink className="h-3.5 w-3.5" aria-hidden />
              <span className="sr-only">(opens in a new tab)</span>
            </a>
          )}

          {claims.length > 0 && (
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-faint">
                Cited by
              </h3>
              <ul className="space-y-1.5">
                {claims.map((c) => (
                  <li key={c.key}>
                    <button
                      type="button"
                      onClick={() => onJumpToClaim(c.key)}
                      className="w-full rounded-md border border-rule bg-paper px-2.5 py-1.5 text-left text-sm hover:border-accent"
                    >
                      <span className="line-clamp-3">{c.text}</span>
                      {c.status && (
                        <span className="mt-0.5 block text-xs text-ink-faint">
                          {STATUS_LABEL[c.status]}
                        </span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
