import { useEffect, useState } from "react";
import { AlertTriangle, Loader2, RotateCcw, SearchX } from "lucide-react";
import type { ApiError } from "../api/client";
import type { IndexStatus, ResearchAnswer } from "../types/api";

export const EXAMPLE_QUESTIONS = [
  "What is retrieval-augmented generation and how does it combine memory with a generator?",
  "How does dense passage retrieval compare with BM25 keyword search?",
  "What problem does reciprocal rank fusion solve in hybrid search?",
  "How do Sentence-BERT embeddings differ from standard BERT for similarity search?",
];

export function InitialState({
  onPick,
  index,
}: {
  onPick: (q: string) => void;
  index: IndexStatus | null;
}) {
  const docs = index?.counts?.documents;
  return (
    <div className="mt-10">
      <h2 className="mb-1 font-serif text-2xl font-semibold">Ask a research question</h2>
      <p className="mb-6 max-w-prose text-ink-soft">
        Answers are written from passages retrieved from the indexed corpus
        {typeof docs === "number" ? ` (${docs} document${docs === 1 ? "" : "s"} currently indexed)` : ""},
        and every claim links back to the passage it relies on.
      </p>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-faint">
        Example questions
      </h3>
      <ul className="space-y-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <li key={q}>
            <button
              type="button"
              onClick={() => onPick(q)}
              className="w-full rounded-lg border border-rule bg-paper-raised px-4 py-3 text-left font-serif text-[1.02rem] leading-snug hover:border-accent"
            >
              {q}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function LoadingState({ decompose }: { decompose: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <div
      role="status"
      aria-live="polite"
      className="mt-10 flex items-start gap-3 rounded-xl border border-rule bg-paper-raised px-5 py-4"
    >
      <Loader2 className="mt-0.5 h-5 w-5 animate-spin text-accent motion-reduce:animate-none" aria-hidden />
      <div>
        <p className="font-medium">Researching…</p>
        <p className="text-sm text-ink-soft">
          The backend is {decompose ? "planning, " : ""}retrieving evidence and writing a grounded
          answer. The first request after startup can take up to a minute while models load.
        </p>
        <p className="mt-1 text-xs tabular-nums text-ink-faint">{elapsed}s elapsed</p>
      </div>
    </div>
  );
}

const TITLES: Record<string, string> = {
  network: "Can’t reach the backend",
  timeout: "The request timed out",
  rate_limit: "Language model rate limit",
  index_unavailable: "Research index unavailable",
  model_unavailable: "Model unavailable",
  invalid_request: "Request rejected",
  server: "The backend reported an error",
  malformed: "Unexpected response",
};

export function ErrorState({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  return (
    <div role="alert" className="mt-10 rounded-xl border border-danger/30 bg-danger-tint px-5 py-4">
      <p className="flex items-center gap-2 font-medium text-danger">
        <AlertTriangle className="h-4 w-4" aria-hidden />
        {TITLES[error.kind] ?? "Something went wrong"}
      </p>
      <p className="mt-1 text-sm text-ink">{error.message}</p>
      {error.retryable && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 inline-flex items-center gap-2 rounded-lg border border-danger/40 bg-paper-raised px-3 py-1.5 text-sm font-medium hover:bg-paper"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden /> Try again
        </button>
      )}
      <p className="mt-2 text-xs text-ink-faint">Your question has been kept.</p>
    </div>
  );
}

export function InsufficientEvidence({ answer }: { answer: ResearchAnswer }) {
  return (
    <section
      className="rounded-xl border border-warn/30 bg-warn-tint px-5 py-4"
      aria-label="Insufficient evidence"
    >
      <p className="flex items-center gap-2 font-medium text-warn">
        <SearchX className="h-4 w-4" aria-hidden /> Not enough evidence to answer reliably
      </p>
      <p className="mt-1 text-sm text-ink">
        The indexed sources did not contain enough relevant material to support a reliable answer, so
        no answer was produced.
      </p>
      {answer.summary && <p className="mt-2 text-sm text-ink-soft">Backend message: {answer.summary}</p>}
      {answer.limitations.length > 0 && (
        <ul className="mt-2 list-disc pl-5 text-sm text-ink-soft">
          {answer.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-xs text-ink-faint">
        Try rephrasing, or add documents to the index (see the README).
      </p>
    </section>
  );
}
