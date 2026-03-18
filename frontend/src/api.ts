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

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "http://localhost:8000";

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

  const res = await fetch(`${API_BASE}/orchestrator/process`, {
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

  const res = await fetch(`${API_BASE}/orchestrator/answer_followup`, {
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

  const res = await fetch(`${API_BASE}/orchestrator/specialist_rounds`, {
    method: "POST",
    body: fd
  });

  return await parseJsonOrThrow<unknown>(res);
}

