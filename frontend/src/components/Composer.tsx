import { useId, useState } from "react";
import { Search, Square } from "lucide-react";

export interface ResearchOptions {
  decompose: boolean;
  evidenceK: number;
}

const MAX_LEN = 1500;
const MIN_LEN = 3;

interface Props {
  value: string;
  onChange: (v: string) => void;
  options: ResearchOptions;
  onOptionsChange: (o: ResearchOptions) => void;
  onSubmit: () => void;
  onCancel: () => void;
  loading: boolean;
  compact?: boolean;
}

export function Composer({
  value,
  onChange,
  options,
  onOptionsChange,
  onSubmit,
  onCancel,
  loading,
  compact = false,
}: Props) {
  const id = useId();
  const [touched, setTouched] = useState(false);
  const trimmed = value.trim();
  const tooShort = trimmed.length < MIN_LEN;
  const tooLong = value.length > MAX_LEN;
  const error = touched && trimmed.length === 0
    ? "Enter a research question."
    : touched && tooShort
      ? `Questions need at least ${MIN_LEN} characters.`
      : tooLong
        ? `Questions are limited to ${MAX_LEN} characters (${value.length}).`
        : null;

  const submit = () => {
    setTouched(true);
    if (loading || tooShort || tooLong) return;
    onSubmit();
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="rounded-xl border border-rule bg-paper-raised shadow-[0_1px_0_rgba(35,33,30,0.04)] focus-within:border-accent"
      aria-label="Research question"
    >
      <label htmlFor={id} className="sr-only">
        Research question
      </label>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={() => setTouched(true)}
        onKeyDown={(e) => {
          // Enter inserts a newline (multiline questions); Ctrl/Cmd+Enter submits.
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            submit();
          }
        }}
        rows={compact ? 2 : 4}
        placeholder="Ask a research question — e.g. How does dense retrieval compare with BM25?"
        disabled={loading}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${id}-err` : `${id}-hint`}
        className="block w-full resize-y rounded-t-xl bg-transparent px-4 py-3 font-serif text-[1.05rem] leading-relaxed text-ink placeholder:text-ink-faint focus:outline-none focus-visible:ring-0 disabled:opacity-60"
      />
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-rule px-3 py-2">
        <label className="flex items-center gap-2 text-sm text-ink-soft">
          <input
            type="checkbox"
            checked={options.decompose}
            disabled={loading}
            onChange={(e) => onOptionsChange({ ...options, decompose: e.target.checked })}
            className="h-4 w-4 accent-[#2f5d62]"
          />
          Break into sub-questions
        </label>
        <label className="flex items-center gap-2 text-sm text-ink-soft">
          Evidence passages
          <select
            value={options.evidenceK}
            disabled={loading}
            onChange={(e) => onOptionsChange({ ...options, evidenceK: Number(e.target.value) })}
            className="rounded-md border border-rule bg-paper-raised px-1.5 py-1 text-sm text-ink"
          >
            {[3, 4, 6, 8, 12].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <div className="ml-auto flex items-center gap-3">
          {loading ? (
            <button
              type="button"
              onClick={onCancel}
              className="inline-flex items-center gap-2 rounded-lg border border-rule-strong px-3.5 py-2 text-sm font-medium text-ink hover:bg-paper-sunk"
            >
              <Square className="h-3.5 w-3.5" aria-hidden /> Cancel
            </button>
          ) : (
            <button
              type="submit"
              className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover"
            >
              <Search className="h-4 w-4" aria-hidden /> Research
            </button>
          )}
        </div>
      </div>
      <p
        id={error ? `${id}-err` : `${id}-hint`}
        className={`px-4 pb-2 text-xs ${error ? "text-danger" : "text-ink-faint"}`}
        role={error ? "alert" : undefined}
      >
        {error ?? "Enter adds a new line · Ctrl/⌘ + Enter to research"}
      </p>
    </form>
  );
}
