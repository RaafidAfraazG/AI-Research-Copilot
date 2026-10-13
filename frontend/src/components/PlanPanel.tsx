import { useState } from "react";
import { ChevronDown, GitBranch } from "lucide-react";
import type { PlanView } from "../lib/answerModel";

interface Coverage {
  subqueries?: number;
  subqueries_with_evidence?: number;
  unique_sources?: number;
  failed_subqueries?: string[];
}

export function PlanPanel({
  plan,
  retrieval,
}: {
  plan: PlanView;
  retrieval: Record<string, unknown> | null;
}) {
  const [open, setOpen] = useState(false);
  const cov = (retrieval ?? {}) as Coverage;
  const multi = plan.subqueries.length > 1;
  const failed = Array.isArray(cov.failed_subqueries) ? cov.failed_subqueries : [];

  return (
    <section className="rounded-xl border border-rule bg-paper-raised" aria-label="Research plan">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left"
      >
        <GitBranch className="h-4 w-4 text-accent" aria-hidden />
        <span className="text-sm font-medium">Research plan</span>
        <span className="text-sm text-ink-soft">
          {multi
            ? `${plan.subqueries.length} sub-questions`
            : plan.requiresDecomposition === false
              ? "single query — decomposition not needed"
              : "single query"}
        </span>
        <ChevronDown
          className={`ml-auto h-4 w-4 text-ink-faint transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        />
      </button>
      {open && (
        <div className="space-y-4 border-t border-rule px-4 py-4 text-sm">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
            {plan.intent && <Fact label="Detected intent" value={plan.intent} />}
            {plan.complexity && <Fact label="Complexity" value={plan.complexity} />}
            {plan.strategy && <Fact label="Strategy" value={plan.strategy} />}
            {typeof cov.subqueries_with_evidence === "number" && typeof cov.subqueries === "number" && (
              <Fact
                label="Sub-questions with evidence"
                value={`${cov.subqueries_with_evidence} of ${cov.subqueries}`}
              />
            )}
            {typeof cov.unique_sources === "number" && (
              <Fact label="Distinct sources" value={String(cov.unique_sources)} />
            )}
          </dl>

          {plan.subqueries.length > 0 && (
            <ol className="space-y-2">
              {plan.subqueries.map((sq, i) => (
                <li key={sq.subquery_id} className="rounded-lg bg-paper-sunk px-3 py-2">
                  <p className="font-medium">
                    <span className="mr-2 text-ink-faint">{i + 1}.</span>
                    {sq.text}
                  </p>
                  {sq.purpose && <p className="mt-0.5 text-ink-soft">{sq.purpose}</p>}
                  {sq.dependencies && sq.dependencies.length > 0 && (
                    <p className="mt-0.5 text-xs text-ink-faint">
                      Depends on: {sq.dependencies.join(", ")}
                    </p>
                  )}
                  {failed.includes(sq.subquery_id) && (
                    <p className="mt-0.5 text-xs text-danger">Retrieval for this sub-question failed.</p>
                  )}
                </li>
              ))}
            </ol>
          )}

          {!multi && (
            <p className="text-ink-soft">
              The question was searched as a single query; no sub-questions were generated.
            </p>
          )}
          {(plan.entities.length > 0 || plan.concepts.length > 0) && (
            <p className="text-ink-soft">
              <span className="font-medium text-ink">Detected terms: </span>
              {[...plan.entities, ...plan.concepts].join(", ")}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-ink-faint">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}
