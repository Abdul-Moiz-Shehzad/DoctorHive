export type BackendModel = "gpt" | "gemini";

export type InitialOrchestratorResponse = {
  case_id: string;
  stage: string;
  gp_response: string;
  next_followup: string | null;
  answered_followups: Array<{ question: string; answer: string }>;
  specialists_required: string[] | null;
};

export type FollowUpResponse = {
  case_id: string;
  stage: string;
  message: string;
  next_followup: string | null;
  answered_followups: Array<{ question: string; answer: string }>;
  specialists_required: string[] | null;
};

function normalizeBaseUrl(x: string) {
  return x.replace(/\/+$/, "");
}

// If `VITE_API_BASE` is unset:
// - in dev, default to "" (use the Vite proxy, same-origin)
// - in prod, default to the local backend (override via VITE_API_BASE for deployments)
const API_BASE = normalizeBaseUrl(
  (
    (import.meta.env.VITE_API_BASE as string | undefined) ??
    (import.meta.env.DEV ? "" : "http://localhost:8000")
  ).trim()
);

function url(path: string) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return API_BASE ? `${API_BASE}${p}` : p;
}

async function fetchOrThrow(input: RequestInfo | URL, init?: RequestInit) {
  try {
    return await fetch(input, init);
  } catch (e) {
    // Browser collapses CORS/mixed-content/DNS/refused-connection into `TypeError: Failed to fetch`.
    const hint =
      `Network error contacting backend.\n` +
      `Tried: ${typeof input === "string" ? input : String(input)}\n\n` +
      `Check:\n` +
      `- backend is running (uvicorn on port 8000)\n` +
      `- VITE_API_BASE points to the correct host (or use Vite proxy)\n` +
      `- no HTTPS->HTTP mixed-content blocking\n` +
      `- CORS allows this origin`;
    throw new Error(e instanceof Error ? `${e.message}\n\n${hint}` : hint);
  }
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const json = await res.json();
      detail = (json?.detail as string) ?? JSON.stringify(json);
    } catch {
      try {
        detail = await res.text();
      } catch {
        // ignore
      }
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export async function processCase(params: {
  message: string;
  model: BackendModel;
  caseId?: string;
  files?: File[];
}): Promise<InitialOrchestratorResponse> {
  const fd = new FormData();
  fd.append("message", params.message);
  fd.append("model", params.model);
  if (params.caseId) fd.append("case_id", params.caseId);
  for (const f of params.files ?? []) fd.append("files", f);

  const res = await fetchOrThrow(url("/orchestrator/process"), {
    method: "POST",
    body: fd
  });

  return await parseJsonOrThrow<InitialOrchestratorResponse>(res);
}

export async function answerFollowup(params: {
  caseId: string;
  answer: string;
}): Promise<FollowUpResponse> {
  const fd = new FormData();
  fd.append("case_id", params.caseId);
  fd.append("answer", params.answer);

  const res = await fetchOrThrow(url("/orchestrator/answer_followup"), {
    method: "POST",
    body: fd
  });

  return await parseJsonOrThrow<FollowUpResponse>(res);
}

export async function runSpecialistRounds(params: {
  caseId: string;
  model: BackendModel;
}): Promise<unknown> {
  const fd = new FormData();
  fd.append("case_id", params.caseId);
  fd.append("model", params.model);

  const res = await fetchOrThrow(url("/orchestrator/specialist_rounds"), {
    method: "POST",
    body: fd
  });

  return await parseJsonOrThrow<unknown>(res);
}

