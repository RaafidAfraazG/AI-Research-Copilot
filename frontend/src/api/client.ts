/**
 * Centralized, typed API client. All AI work happens in the FastAPI backend;
 * this module only transports requests and normalizes errors into user-safe
 * messages (raw backend error text is never shown to the user).
 */
import type {
  HealthResponse,
  IndexStatus,
  ResearchAnswer,
  ResearchAnswerRequest,
} from "../types/api";

export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1"
).replace(/\/+$/, "");

/** Answering can include a cold model load plus an LLM call. */
export const ANSWER_TIMEOUT_MS = 180_000;
const SHORT_TIMEOUT_MS = 8_000;

export type ApiErrorKind =
  | "network"
  | "timeout"
  | "aborted"
  | "rate_limit"
  | "index_unavailable"
  | "model_unavailable"
  | "invalid_request"
  | "server"
  | "malformed";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly code?: string;
  readonly retryable: boolean;

  constructor(kind: ApiErrorKind, message: string, opts: { status?: number; code?: string } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = opts.status;
    this.code = opts.code;
    this.retryable = kind !== "invalid_request" && kind !== "aborted";
  }
}

const RATE_LIMIT_RE = /429|quota|rate[\s_-]?limit|resource[\s_-]?exhausted|too many requests/i;

/** Map an HTTP error response to a user-safe ApiError. Exported for tests. */
export function classifyHttpError(status: number, body: unknown): ApiError {
  const obj = (typeof body === "object" && body !== null ? body : {}) as Record<string, unknown>;
  const code = typeof obj.code === "string" ? obj.code : undefined;
  // Backend AppException bodies ({error, code}) carry client-safe messages.
  const safeMessage = typeof obj.error === "string" ? obj.error : undefined;
  const detail = typeof obj.detail === "string" ? obj.detail : "";

  if (code && code.startsWith("index_")) {
    return new ApiError(
      "index_unavailable",
      `The research index is not ready${safeMessage ? `: ${safeMessage}` : "."}`,
      { status, code },
    );
  }
  if (code && code.startsWith("embedding_")) {
    return new ApiError(
      "model_unavailable",
      "The embedding model could not be loaded, so retrieval is unavailable.",
      { status, code },
    );
  }
  // Stable codes from the sanitizing backend handlers (Phase 10).
  if (code === "llm_rate_limited") {
    return new ApiError("rate_limit", safeMessage ?? "The language model is rate limited.", { status, code });
  }
  if (code === "llm_timeout" || code === "upstream_timeout") {
    return new ApiError("timeout", safeMessage ?? "The request timed out upstream.", { status, code });
  }
  if (
    code === "llm_model_unavailable" ||
    code === "llm_auth_error" ||
    code === "llm_not_configured" ||
    code === "llm_provider_error"
  ) {
    return new ApiError("model_unavailable", safeMessage ?? "The language model is unavailable.", { status, code });
  }
  if (code === "llm_malformed_output") {
    return new ApiError(
      "server",
      safeMessage ?? "The language model returned an incomplete answer. Try again.",
      { status, code },
    );
  }
  if (code === "url_not_allowed" || code === "validation_error") {
    return new ApiError("invalid_request", safeMessage ?? "The request was rejected as invalid.", { status, code });
  }
  if (status === 429 || RATE_LIMIT_RE.test(detail)) {
    return new ApiError(
      "rate_limit",
      "The language model reported a rate limit or quota problem. Wait a moment and try again.",
      { status, code },
    );
  }
  if (status === 422 || status === 400) {
    return new ApiError("invalid_request", "The request was rejected as invalid.", { status, code });
  }
  return new ApiError(
    "server",
    "The backend could not complete this request. Check the backend logs and try again.",
    { status, code },
  );
}

async function request<T>(
  path: string,
  init: RequestInit & { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<T> {
  const { timeoutMs = SHORT_TIMEOUT_MS, signal, ...rest } = init;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort);

  try {
    let res: Response;
    try {
      res = await fetch(`${API_BASE_URL}${path}`, {
        ...rest,
        signal: controller.signal,
        headers: { Accept: "application/json", ...(rest.headers ?? {}) },
      });
    } catch (e) {
      if (timedOut) throw new ApiError("timeout", "The request took too long and was stopped.");
      if (e instanceof DOMException && e.name === "AbortError") {
        throw new ApiError("aborted", "Request cancelled.");
      }
      throw new ApiError(
        "network",
        "Cannot reach the research backend. Make sure it is running and the API URL is correct.",
      );
    }

    let body: unknown = null;
    const text = await res.text();
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = null;
      }
    }
    if (!res.ok) throw classifyHttpError(res.status, body);
    if (body === null) throw new ApiError("malformed", "The backend returned an unreadable response.");
    return body as T;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

/** Minimal runtime guard so a malformed payload becomes an error state, not a crash. */
export function isResearchAnswer(x: unknown): x is ResearchAnswer {
  if (typeof x !== "object" || x === null) return false;
  const a = x as Record<string, unknown>;
  return (
    typeof a.question === "string" &&
    typeof a.summary === "string" &&
    Array.isArray(a.claims) &&
    Array.isArray(a.citations) &&
    Array.isArray(a.sections) &&
    typeof a.verification === "object" &&
    a.verification !== null &&
    typeof a.metadata === "object" &&
    a.metadata !== null
  );
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { signal });
}

export function getIndexStatus(signal?: AbortSignal): Promise<IndexStatus> {
  return request<IndexStatus>("/retrieval/index/status", { signal, timeoutMs: 30_000 });
}

export async function researchAnswer(
  req: ResearchAnswerRequest,
  signal?: AbortSignal,
): Promise<ResearchAnswer> {
  const data = await request<unknown>("/research/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    timeoutMs: ANSWER_TIMEOUT_MS,
    signal,
  });
  if (!isResearchAnswer(data)) {
    throw new ApiError("malformed", "The backend returned an answer in an unexpected format.");
  }
  return data;
}
