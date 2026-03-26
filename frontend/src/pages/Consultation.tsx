import React, { useMemo, useState } from "react";
import { postDoctorHive, type BackendModel, type DoctorHiveResponse } from "../api";
import { PlusCircle, Send, UploadCloud, Stethoscope, FileSearch } from "lucide-react";

const MarkdownText = ({ text }: { text: string }) => {
  if (!text) return null;
  const parts = text.split(/(\*\*.*?\*\*|\*.*?\*|\n)/g);
  return (
    <>
      {parts.map((p, i) => {
        if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>;
        if (p.startsWith('*') && p.endsWith('*')) return <em key={i}>{p.slice(1, -1)}</em>;
        if (p === '\n') return <br key={i} />;
        return <span key={i}>{p}</span>;
      })}
    </>
  );
};

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
  const [orchestrator, setOrchestrator] = useState<DoctorHiveResponse | null>(null);
  const [specialistResult, setSpecialistResult] = useState<unknown>(null);

  const [followupAnswer, setFollowupAnswer] = useState("");
  const [ui, setUi] = useState<UiState>({ kind: "idle" });

  // In the unified endpoint, next_followup handles both GP and Specialists
  const nextQuestion = orchestrator?.next_followup ?? null;
  const specialists = orchestrator?.specialists_required ?? null;

  const canSubmit = message.trim().length > 0 && ui.kind !== "loading";
  const canAnswer =
    !!caseId &&
    !!nextQuestion &&
    followupAnswer.trim().length > 0 &&
    ui.kind !== "loading";

  const stageBadge = useMemo(() => {
    return orchestrator?.stage ?? orchestrator?.stage_after ?? "Intake";
  }, [orchestrator?.stage, orchestrator?.stage_after]);

  function onReset() {
    setCaseId(null);
    setOrchestrator(null);
    setSpecialistResult(null);
    setMessage("");
    setFiles([]);
    setFollowupAnswer("");
    setUi({ kind: "idle" });
  }

  /**
   * Recursive loop that drives the backend state machine forward until 
   * a user action (question answer) is required or the case completes.
   */
  async function runOrchestratorLoop(initialParams: Parameters<typeof postDoctorHive>[0]) {
    try {
      let currentParams = initialParams;
      
      while (true) {
        const res = await postDoctorHive(currentParams);
        
        // Track the current active Case ID
        const newCaseId = res.case_id || currentParams.caseId || caseId;
        if (newCaseId && newCaseId !== caseId) {
           setCaseId(newCaseId);
        }

        const stage = res.stage_after || res.stage;

        // If stage is completely missing, it might be the final recommendation payload 
        // returned directly from chat_with_agent at the very end of transfer_control.
        if (!stage) {
            if (res.agent_name && res.message) {
               setSpecialistResult(res);
               setOrchestrator(prev => prev ? { ...prev, stage: "completed", next_followup: null } : null);
               setUi({ kind: "ready" });
               break;
            } else {
               // Unknown generic payload, stop looping to be safe
               setUi({ kind: "ready" });
               break;
            }
        }

        // Update orchestrator state for the UI to display incrementally
        setOrchestrator(prev => ({
           case_id: newCaseId!,
           stage: stage,
           stage_after: res.stage_after,
           message: res.message,
           gp_response: res.gp_response || prev?.gp_response,
           next_followup: res.next_followup || undefined,
           answered_followups: res.answered_followups || prev?.answered_followups || [],
           specialists_required: res.specialists_required || prev?.specialists_required || [],
        }));

        // STOP CONDITIONS: Stages that require physical User Input
        if (stage === "general_follow_up" || stage === "specialists_follow_up") {
           // BUGFIX: If the AI failed to generate an actual follow-up question here, auto-skip the stage!
           if (!res.next_followup || res.next_followup.trim() === "") {
               setUi({ kind: "loading", label: "Advancing past empty follow-up..." });
               currentParams = { caseId: newCaseId!, model, answer: "skip" };
               continue;
           }

           setUi({ kind: "ready" });
           break;
        }

        // STOP CONDITIONS: Case has naturally concluded or we hit an intended terminus
        if (stage === "completed" || stage === "direct_reply" || res.message === "Case already completed.") {
           // We might receive the final structured object here
           if (res.consensus_winner) {
               setSpecialistResult(res.consensus_winner);
           } else if (res.data && res.data.consensus) {
               setSpecialistResult(res.data);
           } else if (res.message && typeof res.message === "string" && res.message.startsWith("{")) {
               try { setSpecialistResult(JSON.parse(res.message)); } catch { /* ignore */ }
           }
           setUi({ kind: "ready" });
           break;
        }

        // LOOP CONDITION: Automated background stages
        // Examples: initial_round, debate, choice, improved_diagnosis, transfer_control
        setUi({ kind: "loading", label: `Running automated phase: ${stage.replace("_", " ")}...` });
        
        // Recursively trigger the unified endpoint with just the caseId and model 
        // to advance the backend sequence automatically.
        currentParams = { caseId: newCaseId, model };
      }
    } catch (e) {
      setUi({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  async function onStartOrContinue() {
    setUi({ kind: "loading", label: caseId ? "Continuing case..." : "Analyzing patient data..." });
    setSpecialistResult(null);
    setFollowupAnswer("");

    await runOrchestratorLoop({
      message,
      model,
      caseId: caseId ?? undefined,
      files
    });
  }

  async function onAnswerFollowup() {
    if (!caseId) return;
    setUi({ kind: "loading", label: "Transmitting response..." });
    
    await runOrchestratorLoop({
      caseId,
      answer: followupAnswer,
      model
    });

    setFollowupAnswer("");
  }

  return (
    <div className="consultation-page">
      <header className="page-header">
        <div>
          <h1 className="title">Active Consultation</h1>
          <div className="subtitle flex gap-2 items-center">
            {caseId ? <><span className="mono">{caseId}</span> &bull; Stage: <span style={{ textTransform: 'capitalize' }}>{stageBadge.replace(/_/g, " ")}</span></> : "New Session"}
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
                    <pre className="pre" style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                      {orchestrator.gp_response?.trim() ? <MarkdownText text={orchestrator.gp_response} /> : "System evaluating patient data..."}
                    </pre>
                 </div>
               </div>
            ) : (
               <div className="empty-state">
                  Awaiting intake submission...
               </div>
            )}

            {/* Unified User Input for any backend follow-up stage (GP or Specialists) */}
            {nextQuestion && (
               <div className="timeline-event follow-up" style={{ marginTop: '24px' }}>
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

            {/* Fully Responsive Component handling ALL final returned forms of `specialistResult` */}
            {specialistResult && (
              <div className="timeline-event" style={{ marginTop: '24px' }}>
                 <div className="event-badge" style={{ background: 'var(--accent-1)', color: 'white' }}>✓</div>
                 <div className="event-content" style={{ background: "rgba(34, 197, 94, 0.08)", border: "1px solid rgba(34, 197, 94, 0.3)", padding: '20px', borderRadius: '12px' }}>
                    <div className="smallTitle text-accent" style={{ marginBottom: '16px', fontSize: '1.25rem', fontWeight: 600 }}>
                      Final Diagnostic Assessment
                    </div>

                    {/* Direct Recommendation Payload */}
                    {(specialistResult as any).role === "recommendation" && (
                      <div style={{ marginBottom: '20px' }}>
                         <h4 style={{ margin: '0 0 8px 0', color: 'var(--foreground)', fontSize: '1.1rem' }}>
                           Recommendation from: {(specialistResult as any).agent_name}
                         </h4>
                         <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: 'var(--foreground)' }}>
                           <MarkdownText text={String((specialistResult as any).message)} />
                         </div>
                      </div>
                    )}
                    
                    {/* Embedded Result Message if wrapped */}
                    {!(specialistResult as any).consensus && (specialistResult as any).winner && (specialistResult as any).message && (
                      <div style={{ marginBottom: '20px' }}>
                         <h4 style={{ margin: '0 0 8px 0', color: 'var(--foreground)', fontSize: '1.1rem' }}>
                           Final Report: {(specialistResult as any).winner}
                         </h4>
                         <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: 'var(--foreground)' }}>
                           <MarkdownText text={String((specialistResult as any).message)} />
                         </div>
                      </div>
                    )}

                    {/* Complex Consensus Object */}
                    {(specialistResult as any).consensus && (
                      <div style={{ marginBottom: '20px', padding: '16px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                         <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent-1)' }}>Specialist Consensus Overview</h4>
                         <div style={{ display: 'grid', gap: '8px' }}>
                           <div><strong>Lead Specialist:</strong> {(specialistResult as any).consensus.winner}</div>
                           <div><strong>Final Diagnosis:</strong> {(specialistResult as any).consensus.diagnosis}</div>
                           <div style={{ marginTop: '8px', lineHeight: 1.5 }}>
                             <strong>Clinical Rationale:</strong><br/>
                             <MarkdownText text={(specialistResult as any).consensus.explanation} />
                           </div>
                         </div>
                      </div>
                    )}

                    {/* Complex Individual Breakdown */}
                    {(specialistResult as any).improved_diagnosis?.results && (
                      <div style={{ marginTop: '20px' }}>
                         <h4 style={{ margin: '0 0 12px 0', color: 'var(--foreground)' }}>Specialist Breakdown</h4>
                         <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                           {Object.entries((specialistResult as any).improved_diagnosis.results).filter(([_, val]) => val !== null).map(([specialist, data]) => {
                             const sd = data as any;
                             return (
                               <div key={specialist} style={{ padding: '12px', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}>
                                 <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                   <strong style={{ textTransform: 'capitalize' }}>{specialist}</strong>
                                   <span style={{ color: 'var(--accent-2)' }}>Confidence: {sd.confidence}%</span>
                                 </div>
                                 <div style={{ color: 'var(--accent-1)', marginBottom: '4px' }}><strong>Diagnosis:</strong> {sd.diagnosis}</div>
                                 <div style={{ fontSize: '0.95rem', lineHeight: 1.5 }}><MarkdownText text={sd.explanation} /></div>
                               </div>
                             );
                           })}
                         </div>
                      </div>
                    )}

                    {/* Raw string fallback */}
                    {typeof specialistResult === 'string' && (
                        <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                          <MarkdownText text={String(specialistResult)} />
                        </div>
                    )}

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
