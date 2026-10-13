import { BookOpenText, Plus, X } from "lucide-react";
import type { IndexStatus } from "../types/api";
import { API_BASE_URL } from "../api/client";

export type Connection = "checking" | "online" | "offline";

interface Props {
  connection: Connection;
  index: IndexStatus | null;
  currentQuestion: string | null;
  onNew: () => void;
  onClose?: () => void;
}

export function Sidebar({ connection, index, currentQuestion, onNew, onClose }: Props) {
  const dot =
    connection === "online" ? "bg-accent" : connection === "offline" ? "bg-danger" : "bg-ink-faint";
  const label =
    connection === "online" ? "Backend connected" : connection === "offline" ? "Backend unreachable" : "Checking backend…";

  return (
    <div className="flex h-full flex-col bg-paper-sunk">
      <div className="flex items-center gap-2 px-4 py-4">
        <BookOpenText className="h-5 w-5 text-accent" aria-hidden />
        <span className="font-serif text-lg font-semibold">AI Research Copilot</span>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close menu"
            className="ml-auto rounded-md p-1.5 hover:bg-paper"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        )}
      </div>

      <div className="px-3">
        <button
          type="button"
          onClick={onNew}
          className="flex w-full items-center gap-2 rounded-lg border border-rule-strong bg-paper-raised px-3 py-2 text-sm font-medium hover:border-accent"
        >
          <Plus className="h-4 w-4" aria-hidden /> New research
        </button>
      </div>

      <nav className="mt-6 flex-1 overflow-y-auto px-4" aria-label="Research session">
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-faint">
          Current session
        </h2>
        {currentQuestion ? (
          <p className="line-clamp-4 rounded-md bg-paper-raised px-2.5 py-2 font-serif text-sm leading-snug">
            {currentQuestion}
          </p>
        ) : (
          <p className="text-sm text-ink-faint">No question yet.</p>
        )}
        <p className="mt-4 text-xs leading-relaxed text-ink-faint">
          Research history isn&rsquo;t available: the backend does not save or list sessions yet.
        </p>
      </nav>

      <div className="space-y-2 border-t border-rule px-4 py-3 text-xs">
        <p className="flex items-center gap-2 font-medium" role="status">
          <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />
          {label}
        </p>
        {index?.index_present && index.embedding && (
          <p className="leading-relaxed text-ink-soft">
            Index: {index.counts?.documents ?? "?"} docs · {index.counts?.chunks ?? "?"} passages
            <br />
            Embeddings: {index.embedding.model} ({index.embedding.dimension}d)
            {index.embedding.is_mock && (
              <span className="ml-1 font-semibold text-danger">MOCK</span>
            )}
            {!index.compatible && <span className="ml-1 font-semibold text-danger">· incompatible</span>}
          </p>
        )}
        {index && !index.index_present && (
          <p className="text-danger">No usable index. Run the index rebuild (see README).</p>
        )}
        <p className="break-all text-ink-faint">API: {API_BASE_URL}</p>
      </div>
    </div>
  );
}
