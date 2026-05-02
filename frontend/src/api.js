function normalizeBaseUrl(x) {
  return x.replace(/\/+$/, "");
}
const API_BASE = normalizeBaseUrl((process.env.REACT_APP_API_BASE || "http://localhost:8000").trim());
function url(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return API_BASE ? `${API_BASE}${p}` : p;
}
function getToken() {
  return localStorage.getItem('doctorhive_token');
}
function authHeaders() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}
async function fetchOrThrow(input, init) {
  try {
    return await fetch(input, init);
  } catch (e) {
    const hint = `Network error contacting backend.\n` + `Tried: ${typeof input === "string" ? input : String(input)}\n\n` + `Check:\n` + `- backend is running (uvicorn on port 8000)\n` + `- VITE_API_BASE points to the correct host (or use Vite proxy)\n` + `- no HTTPS->HTTP mixed-content blocking\n` + `- CORS allows this origin`;
    throw new Error(e instanceof Error ? `${e.message}\n\n${hint}` : hint);
  }
}
async function parseJsonOrThrow(res) {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const json = await res.json();
      detail = json?.detail ?? JSON.stringify(json);
    } catch {
      try {
        detail = await res.text();
      } catch {
        // ignore
      }}
    throw new Error(`${res.status}: ${detail}`);
  }
  return await res.json();
}

// ── Auth ──────────────────────────────────────────────────────────────────
export async function postLogin({ email, password }) {
  const res = await fetchOrThrow(url("/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return await parseJsonOrThrow(res);
}

export async function postRegister({ username, email, password }) {
  const res = await fetchOrThrow(url("/auth/register"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  });
  return await parseJsonOrThrow(res);
}

export async function postChangePassword({ current_password, new_password }) {
  const res = await fetchOrThrow(url("/auth/change-password"), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ current_password, new_password }),
  });
  return await parseJsonOrThrow(res);
}

// ── Orchestrator ──────────────────────────────────────────────────────────
export async function postDoctorHive(params) {
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
    headers: authHeaders(),
    body: fd
  });
  return await parseJsonOrThrow(res);
}

export async function fetchAllCases(userId) {
  const res = await fetchOrThrow(url(`/orchestrator/cases/${userId}`), {
    method: "GET",
    headers: authHeaders(),
  });
  return await parseJsonOrThrow(res);
}

export async function deleteCase(caseId) {
  const res = await fetchOrThrow(url(`/orchestrator/cases/${caseId}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return await parseJsonOrThrow(res);
}

// ── Chat History ──────────────────────────────────────────────────────────
export async function saveChatHistory({ user_id, case_id, snapshot }) {
  const fd = new FormData();
  fd.append("user_id", String(user_id));
  fd.append("case_id", case_id);
  fd.append("snapshot", JSON.stringify(snapshot));
  const res = await fetchOrThrow(url("/orchestrator/chat_history/save"), {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  return await parseJsonOrThrow(res);
}

export async function fetchChatHistory(user_id) {
  const res = await fetchOrThrow(url(`/orchestrator/chat_history/${user_id}`), {
    method: "GET",
    headers: authHeaders(),
  });
  return await parseJsonOrThrow(res);
}

export async function createUserCaseMapping({ user_id, case_id }) {
  const fd = new FormData();
  fd.append("user_id", String(user_id));
  fd.append("case_id", case_id);
  const res = await fetchOrThrow(url("/orchestrator/user_case"), {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  return await parseJsonOrThrow(res);
}

// ── Patient Profile ───────────────────────────────────────────────────────
export async function fetchProfile(user_id) {
  const res = await fetchOrThrow(url(`/profile/${user_id}`), {
    method: "GET",
    headers: authHeaders(),
  });
  if (res.status === 404) return null;
  return await parseJsonOrThrow(res);
}

export async function saveProfile(data) {
  const res = await fetchOrThrow(url("/profile/save"), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  });
  return await parseJsonOrThrow(res);
}
export async function updatePreferredModel(model) {
  const res = await fetchOrThrow(url('/auth/update-model'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ model }),
  });
  return await parseJsonOrThrow(res);
}
