export type BackendModel = "gpt" | "gemini";

export type DoctorHiveResponse = {
  case_id?: string;
  stage?: string;
  stage_after?: string;
  message?: string | null;
  gp_response?: string;
  next_followup?: string | null;
  answered_followups?: Array<{ question: string; answer: string }>;
  specialists_required?: string[] | null;
  data?: any;
  decision?: string;
  consensus_winner?: any;
  agent_name?: string;
};

export type CaseHistory = {
  case_id: string;
  user_message: string;
  stage: string;
  timestamp: string;
  specialists_required: string[] | null;
  consensus_winner: any;
  answered_followups: Array<{ question: string; answer: string }>;
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

export async function postDoctorHive(params: {
  model: BackendModel;
  caseId?: string | null;
  message?: string;
  answer?: string;
  files?: File[];
  agent_name?: string;
  chat_type?: number;
  user_message?: string;
  consensus_data_json?: string;
}): Promise<DoctorHiveResponse> {
  const fd = new FormData();
  fd.append("model", params.model);
  
  if (params.caseId) fd.append("case_id", params.caseId);
  if (params.message) fd.append("message", params.message);
  if (params.answer) fd.append("answer", params.answer);
  if (params.agent_name) fd.append("agent_name", params.agent_name);
  if (params.chat_type !== undefined) fd.append("chat_type", String(params.chat_type));
  if (params.user_message) fd.append("user_message", params.user_message);
  if (params.consensus_data_json) fd.append("consensus_data_json", params.consensus_data_json);

  for (const f of params.files ?? []) {
    fd.append("files", f);
  }

  const res = await fetchOrThrow(url("/orchestrator/doctorhive"), {
    method: "POST",
    body: fd
  });

  return await parseJsonOrThrow<DoctorHiveResponse>(res);
}

export async function fetchAllCases(): Promise<CaseHistory[]> {
  const res = await fetchOrThrow(url("/orchestrator/cases"), {
    method: "GET"
  });
  return await parseJsonOrThrow<CaseHistory[]>(res);
}

export async function deleteCase(caseId: string): Promise<{message: string}> {
  const res = await fetchOrThrow(url(`/orchestrator/cases/${caseId}`), {
    method: "DELETE"
  });
  return await parseJsonOrThrow<{message: string}>(res);
}
