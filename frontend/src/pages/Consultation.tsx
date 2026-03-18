import React, { useMemo, useState } from "react";
import {
  answerFollowup,
  answerSpecialistFollowup,
  getSpecialistFollowupState,
  processCase,
  runSpecialistRounds,
  type BackendModel,
  type InitialOrchestratorResponse,
  type SpecialistFollowUpState
} from "../api";
import { PlusCircle, Send, UploadCloud, Stethoscope, FileSearch } from "lucide-react";

type UiState =
  | { kind: "idle" }
  | { kind: "loading"; label: string }
  | { kind: "error"; message: string }
  | { kind: "ready" };

function prettyJson(x: unknown) {
  return JSON.stringify(x, null, 2);
}

export default function Consultation() {
  const [model, setModel] = useState<BackendModel>("gpt");
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState<File[]>([]);

  const [caseId, setCaseId] = useState<string | null>(null);
  const [orchestrator, setOrchestrator] =
    useState<InitialOrchestratorResponse | null>(null);
  const [specialistResult, setSpecialistResult] = useState<unknown>(null);

  const [followupAnswer, setFollowupAnswer] = useState("");
  const [specialistFollowup, setSpecialistFollowup] =
    useState<SpecialistFollowUpState | null>(null);
  const [specialistFollowupAnswer, setSpecialistFollowupAnswer] = useState("");
  const [ui, setUi] = useState<UiState>({ kind: "idle" });

  const nextQuestion = orchestrator?.next_followup ?? null;
  const specialists = orchestrator?.specialists_required ?? null;

  const canSubmit = message.trim().length > 0 && ui.kind !== "loading";
  const canAnswer =
    !!caseId &&
    !!nextQuestion &&
    followupAnswer.trim().length > 0 &&
    ui.kind !== "loading";

  const specialistNext = specialistFollowup?.next_followup ?? null;
  const canAnswerSpecialist =
    !!caseId && !!specialistNext && specialistFollowupAnswer.trim().length > 0 && ui.kind !== "loading";

  const stageBadge = useMemo(() => {
    return orchestrator?.stage ?? "Intake";
  }, [orchestrator?.stage]);

  async function onStartOrContinue() {
    setUi({ kind: "loading", label: caseId ? "Continuing case..." : "Analyzing patient data..." });
    setSpecialistResult(null);
    setSpecialistFollowup(null);
    setSpecialistFollowupAnswer("");
    try {
      const res = await processCase({
        message,
        model,
        caseId: caseId ?? undefined,
        files
      });
      setOrchestrator(res);
      setCaseId(res.case_id);
      setFollowupAnswer("");

      // Automatically run specialists if GP has no follow-ups but needs specialists
      if (!res.next_followup && Array.isArray(res.specialists_required) && res.specialists_required.length > 0) {
        setUi({ kind: "loading", label: "Consulting specialist network..." });
        try {
          const specRes = await runSpecialistRounds({ caseId: res.case_id, model });
          setSpecialistResult(specRes);
          // If specialists have follow-ups, expose them in the UI.
          try {
            const s = await getSpecialistFollowupState({ caseId: res.case_id });
            setSpecialistFollowup(s.next_followup ? s : null);
          } catch {
            // ignore; specialist follow-up UI is optional
          }
          setUi({ kind: "ready" });
        } catch (specErr) {
          setUi({ kind: "error", message: specErr instanceof Error ? specErr.message : String(specErr) });
        }
      } else {
        setUi({ kind: "ready" });
      }
    } catch (e) {
      setUi({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  async function onAnswerFollowup() {
    if (!caseId) return;
    setUi({ kind: "loading", label: "Transmitting follow-up..." });
    try {
      const res = await answerFollowup({ caseId, answer: followupAnswer });
      setOrchestrator((prev) => {
        if (!prev) {
          return {
            case_id: res.case_id,
            stage: res.stage,
            gp_response: "",
            next_followup: res.next_followup,
            answered_followups: res.answered_followups,
            specialists_required: res.specialists_required
          };
        }
        return {
          ...prev,
          case_id: res.case_id,
          stage: res.stage,
          next_followup: res.next_followup,
          answered_followups: res.answered_followups,
          specialists_required: res.specialists_required
        };
      });
      setFollowupAnswer("");

      // Automatically run specialists if GP has no further follow-ups
      if (!res.next_followup && Array.isArray(res.specialists_required) && res.specialists_required.length > 0) {
        setUi({ kind: "loading", label: "Consulting specialist network..." });
        try {
          const specRes = await runSpecialistRounds({ caseId, model });
          setSpecialistResult(specRes);
          try {
            const s = await getSpecialistFollowupState({ caseId });
            setSpecialistFollowup(s.next_followup ? s : null);
          } catch {
            // ignore
          }
          setUi({ kind: "ready" });
        } catch (specErr) {
          setUi({ kind: "error", message: specErr instanceof Error ? specErr.message : String(specErr) });
        }
      } else {
        setUi({ kind: "ready" });
      }

    } catch (e) {
      setUi({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  async function onAnswerSpecialistFollowup() {
    if (!caseId) return;
    setUi({ kind: "loading", label: "Transmitting specialist follow-up..." });
    try {
      const res = await answerSpecialistFollowup({ caseId, answer: specialistFollowupAnswer });
      setSpecialistFollowup(res.next_followup ? res : null);
      setSpecialistFollowupAnswer("");

      // When completed, backend returns improved diagnosis + consensus in message.
      if (!res.next_followup && res.message) {
        try {
          setSpecialistResult(JSON.parse(res.message));
        } catch {
          setSpecialistResult(res.message);
        }
      }
      setUi({ kind: "ready" });
    } catch (e) {
      setUi({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  function onReset() {
    setCaseId(null);
    setOrchestrator(null);
    setSpecialistResult(null);
    setMessage("");
    setFiles([]);
    setFollowupAnswer("");
    setSpecialistFollowup(null);
    setSpecialistFollowupAnswer("");
    setUi({ kind: "idle" });
  }

  return (
    <div className="consultation-page">
      <header className="page-header">
        <div>
          <h1 className="title">Active Consultation</h1>
          <div className="subtitle flex gap-2 items-center">
            {caseId ? <><span className="mono">{caseId}</span> &bull; Stage: {stageBadge}</> : "New Session"}
          </div>
        </div>
        <div className="right">
          <button className="secondary" onClick={onReset} disabled={ui.kind === "loading"}>
            <PlusCircle size={18} />
            New Patient
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="card consult-input">
          <div className="cardTitle">
            <Stethoscope size={20} className="text-accent" />
            Clinical Intake
          </div>

          <label className="label">
            Intelligence Model
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as BackendModel)}
              disabled={ui.kind === "loading"}
            >
              <option value="gpt">OpenAI GPT-4o</option>
              <option value="gemini">Google Gemini Flash</option>
            </select>
          </label>

          <label className="label">
            Patient Symptoms & History
            <textarea
              rows={5}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Describe chief complaint, onset, severity, and relevant medical history..."
              disabled={ui.kind === "loading"}
            />
          </label>

          <label className="label">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <UploadCloud size={16} /> Attachments
            </div>
            <input
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png"
              disabled={ui.kind === "loading"}
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
            />
            {files.length > 0 ? (
              <div className="hint" style={{ color: 'var(--accent-1)' }}>{files.length} document(s) attached</div>
            ) : (
              <div className="hint">Supported: PDF, JPG, PNG</div>
            )}
          </label>

          <div className="row mt-4">
            <button onClick={onStartOrContinue} disabled={!canSubmit} style={{ width: '100%' }}>
              <Send size={18} />
              {caseId ? "Update Case File" : "Begin Analysis"}
            </button>
          </div>

          {ui.kind === "loading" ? <div className="status">{ui.label}</div> : null}
          {ui.kind === "error" ? <div className="error">{ui.message}</div> : null}
        </section>

        <section className="card consult-timeline">
          <div className="cardTitle">
            <FileSearch size={20} className="text-accent" />
            Diagnostic Timeline
          </div>

          <div className="timeline-container">
            {orchestrator ? (
               <div className="timeline-event">
                 <div className="event-badge">GP</div>
                 <div className="event-content">
                    <pre className="pre" style={{ margin: 0 }}>
                      {orchestrator.gp_response?.trim() ? orchestrator.gp_response : "No response generated."}
                    </pre>
                 </div>
               </div>
            ) : (
               <div className="empty-state">
                  Awaiting intake submission...
               </div>
            )}

            {nextQuestion && (
               <div className="timeline-event follow-up">
                 <div className="event-badge prompt">?</div>
                 <div className="event-content" style={{ background: 'rgba(168, 85, 247, 0.05)', border: '1px solid rgba(168, 85, 247, 0.2)', padding: '16px', borderRadius: '12px' }}>
                    <div className="smallTitle">Required Follow-up</div>
                    <div className="mono mb-4">{nextQuestion}</div>
                    
                    <input
                      style={{ marginTop: '12px' }}
                      value={followupAnswer}
                      onChange={(e) => setFollowupAnswer(e.target.value)}
                      placeholder="Enter patient's response..."
                      disabled={ui.kind === "loading"}
                    />
                    <button onClick={onAnswerFollowup} disabled={!canAnswer} style={{ marginTop: '12px', padding: '10px 16px', fontSize: '14px' }}>
                      Submit Answer
                    </button>
                 </div>
               </div>
            )}

            {specialistNext && !nextQuestion && (
              <div className="timeline-event follow-up">
                <div className="event-badge spec">★</div>
                <div
                  className="event-content"
                  style={{
                    background: "rgba(34, 197, 94, 0.06)",
                    border: "1px solid rgba(34, 197, 94, 0.25)",
                    padding: "16px",
                    borderRadius: "12px"
                  }}
                >
                  <div className="smallTitle">Specialist Follow-up</div>
                  <div className="mono mb-4">{specialistNext}</div>

                  <input
                    style={{ marginTop: "12px" }}
                    value={specialistFollowupAnswer}
                    onChange={(e) => setSpecialistFollowupAnswer(e.target.value)}
                    placeholder="Enter patient's response..."
                    disabled={ui.kind === "loading"}
                  />
                  <button
                    onClick={onAnswerSpecialistFollowup}
                    disabled={!canAnswerSpecialist}
                    style={{ marginTop: "12px", padding: "10px 16px", fontSize: "14px" }}
                  >
                    Submit Answer
                  </button>
                </div>
              </div>
            )}

            {Array.isArray(specialists) && specialists.length > 0 && !nextQuestion && (
              <div className="timeline-event specialist" style={{ marginTop: '24px' }}>
                 <div className="event-badge spec">★</div>
                 <div className="event-content">
                    <div className="smallTitle">Specialists Indicated</div>
                    <div className="flex gap-2 flex-wrap">
                      {specialists.map(s => (
                        <div key={s} className="chip"><span className="mono">{s}</span></div>
                      ))}
                    </div>
                 </div>
              </div>
            )}
          </div>
        </section>

        <section className="card full mt-4">
          <div className="cardTitle">System Telemetry</div>
          <div className="split">
            <details>
              <summary className="summary">Raw Orchestrator State</summary>
              <pre className="pre">{orchestrator ? prettyJson(orchestrator) : "—"}</pre>
            </details>
            <details open>
              <summary className="summary">Specialist Consensus Output</summary>
              <pre className="pre">{specialistResult ? prettyJson(specialistResult) : "—"}</pre>
            </details>
          </div>
        </section>
      </main>
    </div>
  );
}
