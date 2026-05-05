import React, { useMemo, useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { postDoctorHive, fetchProfile, saveChatHistory, createUserCaseMapping, fetchChatHistory } from "../api";
import { useAuth } from "../context/AuthContext";
import { groupIntoRounds } from "../utils/rounds";
import { PlusCircle, Send, Paperclip, Stethoscope, FileSearch, Terminal, User, Bot, Loader2, Play, ChevronDown, RefreshCw } from "lucide-react";

const MarkdownText = ({ text }) => {
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

function prettyJson(x) {
  return JSON.stringify(x, null, 2);
}

export default function Consultation() {
  const { user } = useAuth();
  const location = useLocation();
  const preferredModel = user?.preferred_model || "gemini";
  const [currentInput, setCurrentInput] = useState("");
  const [files, setFiles] = useState([]);
  
  const [submittedMessage, setSubmittedMessage] = useState("");
  const [submittedFiles, setSubmittedFiles] = useState([]);

  const [caseId, setCaseId] = useState(null);
  const [orchestrator, setOrchestrator] = useState(null);
  const [specialistResult, setSpecialistResult] = useState(null);

  const [ui, setUi] = useState({ kind: "idle" });

  const [xaiEnabled, setXaiEnabled] = useState(false);
  const [xaiLogs, setXaiLogs] = useState([]);

  const chatEndRef = useRef(null);

  const nextQuestion = orchestrator?.next_followup ?? null;
  const specialists = orchestrator?.specialists_required ?? null;

  const isCaseCompleted = orchestrator?.stage === "completed" || specialistResult !== null;
  const canSubmit = currentInput.trim().length > 0 && ui.kind !== "loading" && !isCaseCompleted;
  const isFollowupPhase = !!caseId && !!nextQuestion && !isCaseCompleted;

  const stageBadge = useMemo(() => {
    return orchestrator?.stage ?? orchestrator?.stage_after ?? "Intake";
  }, [orchestrator?.stage, orchestrator?.stage_after]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [orchestrator, specialistResult, xaiLogs, ui]);

  // Handle Resuming Case from History
  useEffect(() => {
    const resumeId = location.state?.resumeCaseId;
    if (resumeId && !caseId && user) {
      setCaseId(resumeId);
      setUi({ kind: "loading", label: "Resuming consultation history..." });
      
      // Attempt to load snapshot from DB first
      fetchChatHistory(user.user_id)
        .then(allHistory => {
          const match = allHistory.find(h => h.case_id === resumeId);
          if (match && match.snapshot) {
            const s = match.snapshot;
            // Migrate answered_followups
            let loadedFollowups = s.answered_followups ?? prev?.answered_followups ?? [];
            loadedFollowups = loadedFollowups.map((qa, i) => {
               if (qa.isSpecialist === undefined) {
                   return { ...qa, isSpecialist: i >= 4 }; // Best-effort migration for old chats
               }
               return qa;
            });

            // Migrate xaiLogs
            let loadedXai = s.xai_logs;
            if (!loadedXai || loadedXai.length === 0) {
               try {
                   loadedXai = JSON.parse(localStorage.getItem(`xai_${resumeId}`) || '[]');
               } catch(e) {}
            }
            loadedXai = (loadedXai || []).map((log, idx) => {
                // Fix old incorrect future stages
                if (log.stage === 'debate' && idx === 0) return { ...log, stage: 'initial_round' };
                if (log.stage === 'specialists_follow_up' && idx === 1) return { ...log, stage: 'debate' };
                if (log.stage === 'choice' || log.stage === 'transfer_control') return { ...log, stage: 'improved_diagnosis' };
                return log;
            });

            setOrchestrator(prev => ({
              ...prev,
              gp_response: s.gp_response ?? prev?.gp_response,
              answered_followups: loadedFollowups,
              specialists_required: s.specialists_required ?? prev?.specialists_required ?? [],
              stage: s.stage ?? prev?.stage,
              stage_after: s.stage_after ?? prev?.stage_after,
              next_followup: s.next_followup ?? prev?.next_followup,
              message: s.message ?? prev?.message,
            }));
            setSpecialistResult(s.specialist_result ?? null);
            setSubmittedMessage(s.submitted_message ?? "");
            setSubmittedFiles(s.submitted_files ?? []);
            setXaiLogs(loadedXai);
          }
          // Continue loop
          runOrchestratorLoop({ caseId: resumeId, model: preferredModel });
        })
        .catch(err => {
          console.error("Failed to fetch history snapshot:", err);
          runOrchestratorLoop({ caseId: resumeId, model: preferredModel });
        });
    }
  }, [location.state, caseId, preferredModel, user]);

  // Persist state to DB whenever it changes
  useEffect(() => {
    if (user && caseId && (orchestrator || specialistResult || submittedMessage || submittedFiles.length > 0)) {
      saveChatHistory({
        user_id: user.user_id,
        case_id: caseId,
        snapshot: {
          gp_response: orchestrator?.gp_response,
          stage: orchestrator?.stage || orchestrator?.stage_after,
          stage_after: orchestrator?.stage_after,
          next_followup: orchestrator?.next_followup,
          specialists_required: orchestrator?.specialists_required,
          answered_followups: orchestrator?.answered_followups,
          specialist_result: specialistResult,
          submitted_message: submittedMessage,
          submitted_files: submittedFiles.map((file) => typeof file === 'string' ? file : file.name),
          xai_logs: xaiLogs,
          message: orchestrator?.message,
        }
      }).catch(console.error);
    }
  }, [user, caseId, orchestrator, specialistResult, submittedMessage, submittedFiles, xaiLogs]);

  function onReset() {
    setCaseId(null);
    setOrchestrator(null);
    setSpecialistResult(null);
    setCurrentInput("");
    setSubmittedMessage("");
    setFiles([]);
    setSubmittedFiles([]);
    setXaiLogs([]);
    setUi({ kind: "idle" });
  }

  async function runOrchestratorLoop(params) {
    try {
      let currentParams = { ...params };
      while (true) {
        const res = await postDoctorHive(currentParams);
        const stage = res.stage || res.stage_after;
        const newCaseId = res.case_id || caseId;

        if (newCaseId && !caseId) {
          setCaseId(newCaseId);
          if (user) {
            createUserCaseMapping({ user_id: user.user_id, case_id: newCaseId }).catch(console.error);
          }
        }

        if (!stage) {
          if (res.agent_name && res.message) {
            setSpecialistResult(res);
            setOrchestrator((prev) => prev ? { ...prev, stage: "completed", next_followup: null } : null);
            setUi({ kind: "ready" });
            break;
          } else {
            setUi({ kind: "ready" });
            break;
          }
        }

        setOrchestrator((prev) => {
          let mergedFollowups = prev?.answered_followups || [];
          if (res.answered_followups) {
            const isSpec = stage === 'specialists_follow_up' || stage === 'improved_diagnosis' || prev?.stage === 'specialists_follow_up';
            const incoming = res.answered_followups.map(qa => ({ ...qa, isSpecialist: isSpec, round: prev?.debate_round_count || 1 }));
            
            const existing = [...mergedFollowups];
            for (const newQa of incoming) {
              if (!existing.some(oldQa => oldQa.question === newQa.question)) {
                existing.push(newQa);
              }
            }
            mergedFollowups = existing;
          }

          return {
            ...res,
            case_id: newCaseId,
            stage: stage,
            stage_after: res.stage_after,
            message: res.message,
            debate_round_count: res.debate_round_count || prev?.debate_round_count || 0,
            gp_response: res.gp_response || prev?.gp_response,
            next_followup: res.next_followup || undefined,
            answered_followups: mergedFollowups,
            specialists_required: res.specialists_required || prev?.specialists_required || []
          };
        });

        if (res.data?.responses) {
          setXaiLogs((prev) => {
            const logStage = res.stage_before || stage;
            const updated = [...prev, { stage: logStage, responses: res.data.responses }];
            if (newCaseId) {
              localStorage.setItem(`xai_${newCaseId}`, JSON.stringify(updated));
            }
            return updated;
          });
        }

        if (stage === "general_follow_up" || stage === "specialists_follow_up") {
          if (!res.next_followup || res.next_followup.trim() === "") {
            setUi({ kind: "loading", label: "Advancing past empty follow-up..." });
            currentParams = { caseId: newCaseId, model: preferredModel, answer: "skip" };
            continue;
          }
          setUi({ kind: "ready" });
          break;
        }

        if (stage === "completed" || stage === "direct_reply" || res.message === "Case already completed.") {
          if (res.consensus_winner) {
            setSpecialistResult(res.consensus_winner);
          } else if (res.data && res.data.consensus) {
            setSpecialistResult(res.data);
          } else if (res.message && typeof res.message === "string" && res.message.startsWith("{")) {
            try {setSpecialistResult(JSON.parse(res.message));} catch {/* ignore */}
          }
          setUi({ kind: "ready" });
          break;
        }

        setUi({ kind: "loading", label: `Running automated phase: ${stage.replace("_", " ")}...` });
        currentParams = { caseId: newCaseId, model: preferredModel };
      }
    } catch (e) {
      setUi({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  async function handleSend() {
    if (!canSubmit) return;

    const rawInput = currentInput;
    setCurrentInput("");

    if (isFollowupPhase) {
      setUi({ kind: "loading", label: "Transmitting response..." });
      await runOrchestratorLoop({
        caseId,
        answer: rawInput,
        model: preferredModel
      });
    } else {
      setSubmittedMessage(rawInput);
      setSubmittedFiles(files);
      setUi({ kind: "loading", label: caseId ? "Continuing case..." : "Analyzing patient data..." });
      setSpecialistResult(null);

      // Prepend medical profile context if starting a NEW case
      let finalMessage = rawInput;
      if (!caseId && user) {
        try {
          const profile = await fetchProfile(user.user_id);
          if (profile) {
            const historyStr = [
              profile.age ? `Age: ${profile.age}` : '',
              profile.gender ? `Gender: ${profile.gender}` : '',
              profile.blood_type ? `Blood Type: ${profile.blood_type}` : '',
              profile.conditions?.length ? `Conditions: ${profile.conditions.join(', ')}` : '',
              profile.allergies?.length ? `Allergies: ${profile.allergies.join(', ')}` : '',
              profile.medications?.length ? `Medications: ${profile.medications.join(', ')}` : '',
            ].filter(Boolean).join('; ');
            
            if (historyStr) {
              finalMessage = `[System Note: Patient Medical History (may be outdated, please verify if relevant): ${historyStr}]\n\nPatient Complaint: ${rawInput}`;
            }
          }
        } catch (err) {
          console.warn("Could not fetch profile for context:", err);
        }
      }

      await runOrchestratorLoop({
        message: finalMessage,
        model: preferredModel,
        caseId: caseId ?? undefined,
        files
      });
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const chatNodes = [];

  if (submittedMessage) {
    chatNodes.push(
      <div key="init-msg" className="chat-bubble-wrapper user">
        <div className="chat-bubble user">
          <div style={{ whiteSpace: "pre-wrap" }}>
            {(() => {
              let text = submittedMessage || "";
              text = text.replace(/\[System Note:.*?\]/gi, '');
              text = text.replace(/Patient Complaint:/gi, '');
              return text.trim();
            })()}
          </div>
          {submittedFiles.length > 0 && (
            <div className="mt-4 pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.1)', fontSize: '0.9em', color: 'var(--text-muted)' }}>
              <Paperclip size={14} className="inline mr-2" />
              {submittedFiles.length} file(s) attached
            </div>
          )}
        </div>
      </div>
    );
  }

  if (orchestrator?.gp_response?.trim()) {
    chatNodes.push(
      <div key="gp-resp" className="chat-bubble-wrapper ai">
        <div className="chat-bubble ai ai-box">
          <div className="chat-bubble-title">
            <Stethoscope size={18} /> GP Analysis
          </div>
          <MarkdownText text={orchestrator.gp_response} />
        </div>
      </div>
    );
  }

  let insertedSpecialistDivider = false;

  const insertSpecialistDivider = () => {
    insertedSpecialistDivider = true;
    chatNodes.push(
      <div key="sys-msg-transition" className="chat-bubble-wrapper ai" style={{ justifyContent: 'center', margin: '8px 0' }}>
        <div className="chat-bubble ai" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--text-muted)', fontSize: '0.85em', padding: '8px 16px', borderRadius: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Terminal size={14} />
          All GP follow-up questions answered. Forwarding to specialists.
        </div>
      </div>
    );
    chatNodes.push(
      <div key="specs-transition" className="chat-bubble-wrapper ai">
        <div className="chat-bubble ai ai-box">
          <div className="chat-bubble-title">
            <FileSearch size={18} /> Specialists Indicated
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}>
            {specialists.map(s => <div key={s} className="chip"><span className="mono">{s}</span></div>)}
          </div>
        </div>
      </div>
    );
  };

  if (orchestrator?.answered_followups) {
    orchestrator.answered_followups.forEach((qa, i) => {
      if (qa.answer === "skip") return;

      const isSpecialistPhase = qa.isSpecialist === true;
      
      if (isSpecialistPhase && !insertedSpecialistDivider && Array.isArray(specialists) && specialists.length > 0) {
        insertSpecialistDivider();
      }

      chatNodes.push(
        <div key={`qa-q-${i}`} className="chat-bubble-wrapper ai">
          <div className="chat-bubble ai ai-box">
            <div className="chat-bubble-title">
              <Bot size={18} /> {isSpecialistPhase ? "Specialist Follow-up" : "GP Follow-up"}
            </div>
            <MarkdownText text={qa.question} />
          </div>
        </div>
      );
      chatNodes.push(
        <div key={`qa-a-${i}`} className="chat-bubble-wrapper user">
          <div className="chat-bubble user">
            <div style={{ whiteSpace: "pre-wrap" }}>{qa.answer}</div>
          </div>
        </div>
      );
    });
  }

  const isGPPhaseOver = orchestrator?.stage && !['initial_round', 'general_follow_up'].includes(orchestrator.stage);

  if (!insertedSpecialistDivider && Array.isArray(specialists) && specialists.length > 0 && isGPPhaseOver) {
    insertSpecialistDivider();
  }

  if (orchestrator?.message && orchestrator.message !== "Answer recorded." && !orchestrator.message.includes("Patient answer required") && !orchestrator.message.includes("Forwarding to specialists") && !specialistResult) {
    chatNodes.push(
      <div key="sys-msg" className="chat-bubble-wrapper ai" style={{ justifyContent: 'center', margin: '8px 0' }}>
        <div className="chat-bubble ai" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--text-muted)', fontSize: '0.85em', padding: '8px 16px', borderRadius: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Terminal size={14} />
          {orchestrator.message}
        </div>
      </div>
    );
  }

  if (nextQuestion) {
    chatNodes.push(
      <div key="next-q" className="chat-bubble-wrapper ai">
        <div className="chat-bubble ai ai-box" style={{ borderColor: 'var(--accent-primary)', background: 'rgba(78, 79, 235, 0.05)' }}>
          <div className="chat-bubble-title">
            <Bot size={18} /> Required Follow-up
          </div>
          <MarkdownText text={nextQuestion} />
        </div>
      </div>
    );
  }



  if (specialistResult) {
    chatNodes.push(
      <div key="final" className="chat-bubble-wrapper ai">
        <div className="chat-bubble ai ai-box" style={{ background: 'var(--success-bg)', borderColor: 'var(--success-text)' }}>
          <div className="chat-bubble-title" style={{ color: 'var(--success-text)', fontSize: '1.2rem' }}>
            ✓ Final Diagnostic Assessment
          </div>
          
          {specialistResult.role === "recommendation" && (
            <div className="mt-4">
              <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-primary)' }}>Recommendation from: {specialistResult.agent_name}</h4>
              <MarkdownText text={String(specialistResult.message)} />
            </div>
          )}

          {!specialistResult.consensus && specialistResult.winner && specialistResult.message && (
            <div className="mt-4">
              <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-primary)' }}>Final Report: {specialistResult.winner}</h4>
              <MarkdownText text={String(specialistResult.message)} />
            </div>
          )}

          {specialistResult.consensus && (
            <div className="mt-4" style={{ background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '8px' }}>
              <h4 style={{ margin: '0 0 12px 0', color: 'var(--success-text)' }}>Specialist Consensus Overview</h4>
              <div style={{ display: 'grid', gap: '8px' }}>
                <div><strong>Lead Specialist:</strong> <span style={{ textTransform: 'capitalize' }}>{specialistResult.consensus.winner}</span></div>
                <div><strong>Final Diagnosis:</strong> {specialistResult.consensus.diagnosis}</div>
                <div className="mt-4">
                  <strong>Clinical Rationale:</strong><br />
                  <div className="mt-4 text-muted"><MarkdownText text={specialistResult.consensus.explanation} /></div>
                </div>
              </div>
            </div>
          )}

          {specialistResult.improved_diagnosis?.results && (
            <div className="mt-8">
              <div style={{ margin: '16px 0', borderBottom: '1px solid rgba(255,255,255,0.1)' }} />
              <h4 style={{ margin: '0 0 12px 0', color: 'var(--text-primary)' }}>Final Specialist Reports</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {Object.entries(specialistResult.improved_diagnosis.results).filter(([_, val]) => val !== null).map(([specialist, sd]) => (
                  <div key={specialist} style={{ padding: '12px', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', background: 'rgba(0,0,0,0.1)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <strong style={{ textTransform: 'capitalize' }}>{specialist}</strong>
                      <span style={{ color: 'var(--success-text)' }}>Confidence: {sd.confidence}%</span>
                    </div>
                    <div style={{ color: 'var(--success-text)', marginBottom: '8px' }}><strong>Diagnosis:</strong> {sd.diagnosis}</div>
                    <div style={{ fontSize: '0.95rem' }}><MarkdownText text={sd.explanation} /></div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {typeof specialistResult === 'string' && (
             <div className="mt-4"><MarkdownText text={String(specialistResult)} /></div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="consultation-page chat-interface">
        {xaiEnabled && (
          <div className="xai-drawer" style={{ 
            position: 'fixed', right: 0, top: 0, bottom: 0, width: '400px', 
            background: 'var(--bg-card)', borderLeft: '1px solid var(--border-color)',
            boxShadow: '-4px 0 20px rgba(0,0,0,0.1)', zIndex: 100,
            display: 'flex', flexDirection: 'column'
          }}>
            <div style={{ padding: '20px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', fontSize: '16px' }}><Terminal size={18} /> Explainable AI (XAI)</h3>
              <button onClick={() => setXaiEnabled(false)} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '18px' }}>✕</button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              
              <div style={{ padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <h4 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}><Bot size={14}/> Orchestrator State</h4>
                <div style={{ fontSize: '13px', display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}><span>Current Phase:</span> <strong>{orchestrator?.stage || 'Init'}</strong></div>
                <div style={{ fontSize: '13px', display: 'flex', justifyContent: 'space-between' }}><span>Debate Rounds:</span> <strong>{orchestrator?.debate_round_count || 0} / 2</strong></div>
              </div>

              {orchestrator?.specialists_required && orchestrator.specialists_required.length > 0 && (
                <div style={{ padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                  <h4 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}><FileSearch size={14}/> Agents Invoked</h4>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {orchestrator.specialists_required.map(s => <div key={s} className="chip"><span className="mono">{s}</span></div>)}
                  </div>
                </div>
              )}

              {groupIntoRounds(xaiLogs).map((round) => (
                <div key={round.roundNum} style={{ marginBottom: '16px' }}>
                  <h4 style={{ margin: '0 0 12px 0', color: 'var(--text-primary)', fontSize: '14px', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>Specialist Round {round.roundNum}</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {round.logs.map((log, i) => {
                      const stageLabel = log.stage === 'specialists_follow_up' ? 'improved diagnosis' : log.stage.replace(/_/g, ' ');
                      return (
                        <div key={i} style={{ padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                          <h4 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', textTransform: 'capitalize' }}><Stethoscope size={14}/> Stage: {stageLabel}</h4>
                          {Object.entries(log.responses).map(([agent, data]) => (
                            <div key={agent} style={{ marginTop: '8px', padding: '8px', background: 'var(--bg-primary)', borderRadius: '4px', border: '1px solid var(--border-color)' }}>
                              <div style={{ fontWeight: 600, textTransform: 'capitalize', fontSize: '12px', color: 'var(--accent-primary)', marginBottom: '4px' }}>{agent}</div>
                              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                                {data === null ? (
                                  <span style={{ fontStyle: 'italic', opacity: 0.7 }}>No response / Not involved</span>
                                ) : typeof data === 'object' ? (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                    {data.confidence && (
                                      <div><strong style={{ color: 'var(--success-text)' }}>Confidence:</strong> {data.confidence}%</div>
                                    )}
                                    {data.diagnosis && (
                                      <div><strong style={{ color: 'var(--text-primary)' }}>Diagnosis:</strong> {data.diagnosis}</div>
                                    )}
                                    {data.explanation && (
                                      <div><strong style={{ color: 'var(--text-primary)' }}>Rationale:</strong><div style={{ marginTop: '2px', maxHeight: '100px', overflowY: 'auto', paddingRight: '4px' }}>{data.explanation}</div></div>
                                    )}
                                    {data.follow_ups && Array.isArray(data.follow_ups) && (
                                      <div><strong style={{ color: 'var(--accent-glow)' }}>Follow-ups requested:</strong>
                                        <ul style={{ margin: '4px 0 0 0', paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                          {data.follow_ups.map((q, idx) => <li key={idx}>{q}</li>)}
                                        </ul>
                                      </div>
                                    )}
                                  </div>
                                ) : (
                                  <div style={{ whiteSpace: 'pre-wrap', maxHeight: '100px', overflowY: 'auto' }}>{String(data)}</div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}

              <div style={{ padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                 <h4 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}><Terminal size={14}/> Raw Telemetry</h4>
                 <pre style={{ fontSize: '11px', whiteSpace: 'pre-wrap', color: 'var(--text-muted)' }}>
                    {JSON.stringify(orchestrator, null, 2)}
                 </pre>
              </div>
            </div>
          </div>
        )}
      <header className="page-header" style={{ marginBottom: '20px' }}>
        <div>
          <h1 className="title">Active Consultation</h1>
          <div className="subtitle flex gap-2 items-center">
            {caseId ? <><span className="mono">{caseId}</span> &bull; Stage: <span style={{ textTransform: 'capitalize' }}>{stageBadge.replace(/_/g, " ")}</span></> : "New Session"}
          </div>
        </div>
      </header>

      {ui.kind === "error" && (
        <div className="error mb-4">{ui.message}</div>
      )}

      <main className="chat-history">
        {chatNodes.length === 0 && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            <Stethoscope size={48} style={{ opacity: 0.2, marginBottom: '20px' }} />
            <h2 style={{ fontSize: '24px', color: 'var(--text-primary)', marginBottom: '8px' }}>DoctorHive Assistant</h2>
            <p>Describe the patient's chief complaint, onset, and relevant medical history to begin.</p>
          </div>
        )}
        
        {chatNodes}

        {ui.kind === "loading" && (
          <div className="chat-bubble-wrapper ai">
            <div className="chat-bubble ai" style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-muted)' }}>
              <Loader2 size={18} className="animate-spin" />
              {ui.label}
            </div>
          </div>
        )}
        
        <div ref={chatEndRef} />
      </main>

      <div className="chat-input-wrapper">
        <div className={`chat-input-container ${ui.kind === "loading" ? "disabled" : ""}`}>
          <div className="chat-input-main">
            <textarea
              rows={Math.min(5, currentInput.split('\n').length)}
              value={currentInput}
              onChange={(e) => setCurrentInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isCaseCompleted ? "Case is closed and cannot be modified." : isFollowupPhase ? "Type the patient's response..." : "Describe the patient's condition..."}
              disabled={ui.kind === "loading" || isCaseCompleted}
              style={{ flex: 1 }}
            />
            <button 
              className="chat-send-btn" 
              onClick={handleSend} 
              disabled={!canSubmit}
              style={{ background: canSubmit ? 'var(--accent-primary)' : 'var(--bg-tertiary)', color: canSubmit ? '#fff' : 'var(--text-muted)' }}
            >
              <Send size={18} style={{ marginLeft: '2px' }} />
            </button>
          </div>
          
          <div className="chat-controls">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              {!isFollowupPhase && (
                <label style={{ position: 'relative' }}>
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.jpg,.jpeg,.png"
                    disabled={ui.kind === "loading"}
                    onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
                    style={{ position: 'absolute', width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }}
                  />
                  <div className="chat-attach-btn" title="Attach files">
                    <Paperclip size={20} style={{ color: files.length > 0 ? 'var(--accent-primary)' : 'inherit' }} />
                    {files.length > 0 && <span style={{ fontSize: '12px', marginLeft: '6px', color: 'var(--accent-primary)' }}>{files.length}</span>}
                  </div>
                </label>
              )}
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <button
                onClick={() => setXaiEnabled(!xaiEnabled)}
                disabled={ui.kind === "loading"}
                title="Toggle Explainable AI Reasoning"
                style={{
                  background: xaiEnabled ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                  color: xaiEnabled ? '#10b981' : 'var(--text-muted)',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '0 10px',
                  height: '30px',
                  borderRadius: '16px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  transition: 'all 0.2s',
                  boxShadow: 'none'
                }}
              >
                <Terminal size={16} />
                <span style={{ fontWeight: 600 }}>XAI</span>
              </button>

            </div>
          </div>
        </div>
      </div>
    </div>
  );
}