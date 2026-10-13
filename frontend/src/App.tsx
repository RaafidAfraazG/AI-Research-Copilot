import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Menu, PanelRight } from "lucide-react";
import { ApiError, getHealth, getIndexStatus, researchAnswer } from "./api/client";
import type { Citation, IndexStatus, ResearchAnswer } from "./types/api";
import { buildSections, readPlan } from "./lib/answerModel";
import { Sidebar, type Connection } from "./components/Sidebar";
import { Composer, type ResearchOptions } from "./components/Composer";
import { PlanPanel } from "./components/PlanPanel";
import { AnswerView } from "./components/AnswerView";
import { EvidencePanel } from "./components/EvidencePanel";
import { ErrorState, InitialState, InsufficientEvidence, LoadingState } from "./components/States";

type Phase = "idle" | "loading" | "success" | "error";

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setMatches(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return matches;
}

export default function App() {
  const [query, setQuery] = useState("");
  const [options, setOptions] = useState<ResearchOptions>({ decompose: true, evidenceK: 6 });
  const [phase, setPhase] = useState<Phase>("idle");
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [answer, setAnswer] = useState<ResearchAnswer | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const [connection, setConnection] = useState<Connection>("checking");
  const [index, setIndex] = useState<IndexStatus | null>(null);

  const [menuOpen, setMenuOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null);
  const [highlightKey, setHighlightKey] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const wide = useMediaQuery("(min-width: 1024px)");

  // Backend liveness (real /health) — re-checked periodically.
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        await getHealth();
        if (alive) setConnection("online");
      } catch {
        if (alive) setConnection("offline");
      }
    };
    void check();
    const t = setInterval(check, 30_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  // Index status (real /retrieval/index/status) once the backend is reachable.
  const refreshIndex = useCallback(async () => {
    try {
      setIndex(await getIndexStatus());
    } catch {
      setIndex(null);
    }
  }, []);
  useEffect(() => {
    if (connection === "online") void refreshIndex();
  }, [connection, refreshIndex]);

  const sections = useMemo(() => (answer ? buildSections(answer) : []), [answer]);
  const plan = useMemo(() => (answer ? readPlan(answer) : null), [answer]);

  const run = useCallback(
    async (text: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setPhase("loading");
      setError(null);
      setSubmitted(text);
      setSelectedEvidence(null);
      setPanelOpen(false);
      try {
        const result = await researchAnswer(
          {
            query: text,
            enable_decomposition: options.decompose,
            evidence_k: options.evidenceK,
          },
          controller.signal,
        );
        setAnswer(result);
        setPhase("success");
        if (wide && (result.evidence?.length ?? 0) > 0) setPanelOpen(true);
      } catch (e) {
        if (e instanceof ApiError && e.kind === "aborted") {
          setPhase(answer ? "success" : "idle");
          return;
        }
        setError(
          e instanceof ApiError ? e : new ApiError("server", "An unexpected error occurred."),
        );
        setPhase("error");
      }
    },
    [options, wide, answer],
  );

  const submit = () => void run(query.trim());
  const cancel = () => abortRef.current?.abort();
  const newResearch = () => {
    abortRef.current?.abort();
    setPhase("idle");
    setAnswer(null);
    setError(null);
    setSubmitted(null);
    setQuery("");
    setPanelOpen(false);
    setSelectedEvidence(null);
    setMenuOpen(false);
  };

  const openCitation = (c: Citation) => {
    setSelectedEvidence(c.evidence_id);
    setPanelOpen(true);
  };
  const jumpToClaim = (key: string) => {
    setHighlightKey(key);
    if (!wide) setPanelOpen(false);
    requestAnimationFrame(() =>
      document.getElementById(`claim-${key}`)?.scrollIntoView({ behavior: "smooth", block: "center" }),
    );
  };

  const showAnswer = (phase === "success" || (phase === "loading" && answer)) && answer;
  const evidenceCount = Math.max(answer?.evidence?.length ?? 0, answer?.citations.length ?? 0);
  const hasWorkspace = phase !== "idle";
  const panelVisible = panelOpen && answer && phase !== "loading";

  return (
    <div className="flex h-dvh overflow-hidden">
      {/* Sidebar: docked on md+, off-canvas on small screens */}
      <aside className="hidden w-64 shrink-0 border-r border-rule md:block" aria-label="Sidebar">
        <Sidebar
          connection={connection}
          index={index}
          currentQuestion={submitted}
          onNew={newResearch}
        />
      </aside>
      {menuOpen && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Menu">
          <button
            type="button"
            aria-label="Close menu"
            className="absolute inset-0 bg-ink/40"
            onClick={() => setMenuOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-72 max-w-[85%] shadow-xl">
            <Sidebar
              connection={connection}
              index={index}
              currentQuestion={submitted}
              onNew={newResearch}
              onClose={() => setMenuOpen(false)}
            />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1">
        <main className="min-w-0 flex-1 overflow-y-auto" id="main">
          <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-rule bg-paper/95 px-4 py-2 backdrop-blur md:hidden">
            <button
              type="button"
              onClick={() => setMenuOpen(true)}
              aria-label="Open menu"
              className="rounded-md p-1.5 hover:bg-paper-sunk"
            >
              <Menu className="h-5 w-5" aria-hidden />
            </button>
            <span className="font-serif font-semibold">AI Research Copilot</span>
          </div>

          <div className="mx-auto w-full max-w-3xl px-4 pb-24 pt-6 sm:px-6 sm:pt-10">
            {connection === "offline" && phase === "idle" && (
              <p
                role="status"
                className="mb-4 rounded-lg border border-danger/30 bg-danger-tint px-4 py-2 text-sm text-danger"
              >
                The research backend is not reachable. Start it (see README) and this page will
                reconnect automatically.
              </p>
            )}

            <Composer
              value={query}
              onChange={setQuery}
              options={options}
              onOptionsChange={setOptions}
              onSubmit={submit}
              onCancel={cancel}
              loading={phase === "loading"}
              compact={hasWorkspace}
            />

            {phase === "idle" && <InitialState onPick={setQuery} index={index} />}
            {phase === "loading" && <LoadingState decompose={options.decompose} />}
            {phase === "error" && error && <ErrorState error={error} onRetry={submit} />}

            {showAnswer && phase === "success" && (
              <div className="mt-8 space-y-6">
                <div className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-semibold uppercase tracking-wider text-ink-faint">
                      Question
                    </p>
                    <h1 className="font-serif text-2xl font-semibold leading-snug">
                      {answer.question}
                    </h1>
                  </div>
                  {evidenceCount > 0 && !panelVisible && (
                    <button
                      type="button"
                      onClick={() => setPanelOpen(true)}
                      className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-rule-strong px-3 py-1.5 text-sm font-medium hover:bg-paper-sunk"
                    >
                      <PanelRight className="h-4 w-4" aria-hidden /> Evidence ({evidenceCount})
                    </button>
                  )}
                </div>

                {plan && <PlanPanel plan={plan} retrieval={answer.metadata.retrieval} />}

                {answer.insufficient_evidence ? (
                  <InsufficientEvidence answer={answer} />
                ) : (
                  <AnswerView
                    answer={answer}
                    sections={sections}
                    selectedEvidenceId={selectedEvidence}
                    highlightedClaimKey={highlightKey}
                    onOpenCitation={openCitation}
                  />
                )}

                <footer className="border-t border-rule pt-3 text-xs text-ink-faint">
                  Generated by {answer.metadata.provider} · {answer.metadata.model} ·{" "}
                  {(answer.metadata.total_latency_ms / 1000).toFixed(1)}s
                  {answer.metadata.total_tokens != null
                    ? ` · ${answer.metadata.total_tokens} tokens`
                    : ""}
                  {answer.metadata.provider === "mock" && (
                    <span className="ml-1 font-semibold text-danger">
                      · MOCK PROVIDER — not a real model answer
                    </span>
                  )}
                </footer>
              </div>
            )}
          </div>
        </main>

        {/* Evidence: docked column on lg+, modal sheet below */}
        {panelVisible && wide && (
          <div className="w-[26rem] shrink-0 border-l border-rule">
            <EvidencePanel
              answer={answer}
              sections={sections}
              selectedEvidenceId={selectedEvidence}
              onSelect={setSelectedEvidence}
              onClose={() => setPanelOpen(false)}
              onJumpToClaim={jumpToClaim}
              overlay={false}
            />
          </div>
        )}
      </div>
      {panelVisible && !wide && (
        <div className="fixed inset-0 z-30">
          <button
            type="button"
            aria-label="Close evidence panel"
            className="absolute inset-0 bg-ink/40"
            onClick={() => setPanelOpen(false)}
          />
          <div className="absolute inset-x-0 bottom-0 top-16 overflow-hidden rounded-t-2xl shadow-xl sm:inset-y-0 sm:left-auto sm:right-0 sm:top-0 sm:w-[26rem] sm:rounded-none">
            <EvidencePanel
              answer={answer}
              sections={sections}
              selectedEvidenceId={selectedEvidence}
              onSelect={setSelectedEvidence}
              onClose={() => setPanelOpen(false)}
              onJumpToClaim={jumpToClaim}
              overlay
            />
          </div>
        </div>
      )}
    </div>
  );
}
