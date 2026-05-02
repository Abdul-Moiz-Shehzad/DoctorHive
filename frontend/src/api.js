function normalizeBaseUrl(x) {
  return x.replace(/\/+$/, "");
}
const API_BASE = normalizeBaseUrl((process.env.REACT_APP_API_BASE || "http://localhost:8000").trim());
function url(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return API_BASE ? `${API_BASE}${p}` : p;
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
    body: fd
  });
  return await parseJsonOrThrow(res);
}
export async function fetchAllCases() {
  const res = await fetchOrThrow(url("/orchestrator/cases"), {
    method: "GET"
  });
  return await parseJsonOrThrow(res);
}
export async function deleteCase(caseId) {
  const res = await fetchOrThrow(url(`/orchestrator/cases/${caseId}`), {
    method: "DELETE"
  });
  return await parseJsonOrThrow(res);
}