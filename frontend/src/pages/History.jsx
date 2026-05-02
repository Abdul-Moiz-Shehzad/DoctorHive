import React, { useEffect, useState } from 'react';
import { fetchAllCases, deleteCase } from '../api';
import { Clock, Users, ChevronDown, ChevronUp, Trash2, Terminal } from 'lucide-react';

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
    </>);

};

export default function History() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [xaiEnabled, setXaiEnabled] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchAllCases();
        setCases(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const handleDeleteCase = async (caseId) => {
    // Backup and update UI optimistically for instant feedback
    const previousCases = [...cases];
    setCases((c) => c.filter((x) => x.case_id !== caseId));

    try {
      await deleteCase(caseId);
    } catch (e) {
      console.error("Failed to delete case:", e);
      // Revert on failure
      setCases(previousCases);
      setError("Successfully requested deletion but a connectivity error occurred. Try refreshing.");
    }
  };

  return (
    <div className="history-page">
      <header className="page-header">
        <div>
          <h1 className="title">Patient History</h1>
          <p className="subtitle">Review all past and ongoing consultations.</p>
        </div>
        <div className="right">
          <button
            className={xaiEnabled ? "" : "secondary"}
            onClick={() => setXaiEnabled(!xaiEnabled)}
            title="Toggle Explainable AI Reasoning"
            style={xaiEnabled ? { background: '#10b981', color: '#fff', border: 'none' } : {}}>
            
            <Terminal size={18} />
             XAI Mode {xaiEnabled ? "ON" : "OFF"}
          </button>
        </div>
      </header>

      {loading && <div className="status mt-8">Loading history records...</div>}
      {error && <div className="error mt-8">{error}</div>}

      {!loading && !error && cases.length === 0 &&
      <div className="empty-state mt-8">No consultation history found. Begin your first analysis!</div>
      }

      <div className="cases-list" style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '24px' }}>
        {cases.map((c) => {
          const isExpanded = expandedId === c.case_id;

          let xaiLogs = [];
          if (isExpanded && xaiEnabled) {
            const xaiLogsRaw = localStorage.getItem(`xai_${c.case_id}`);
            xaiLogs = xaiLogsRaw ? JSON.parse(xaiLogsRaw) : [];
          }

          const statusColors = {
            completed: 'var(--success-text)',
            init: 'var(--accent-1)',
            general_follow_up: 'var(--accent-2)',
            specialists_follow_up: 'var(--accent-2)'
          };
          const badgeColor = statusColors[c.stage] || 'var(--foreground)';

          return (
            <div key={c.case_id} className="card history-card" style={{ cursor: 'pointer', transition: 'all 0.2sease' }} onClick={() => setExpandedId(isExpanded ? null : c.case_id)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                 <div>
                   <div className="mono" style={{ fontSize: '0.85rem', color: 'var(--accent-1)', marginBottom: '8px' }}>{c.case_id}</div>
                   <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--foreground)' }}>
                     {c.user_message.length > 80 ? c.user_message.substring(0, 80) + '...' : c.user_message}
                   </div>
                   <div style={{ display: 'flex', gap: '16px', marginTop: '12px', fontSize: '0.9rem', color: 'var(--muted)' }}>
                     <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                       <Clock size={14} /> {c.timestamp ? new Date(c.timestamp).toLocaleString() : 'Unknown Date'}
                     </div>
                     <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                       <Users size={14} /> {c.specialists_required?.length || 0} Specialists
                     </div>
                   </div>
                 </div>
                 <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                   <div className="chip" style={{ background: 'rgba(255,255,255,0.05)', color: badgeColor, border: `1px solid ${badgeColor}40`, textTransform: 'capitalize' }}>
                     {c.stage.replace(/_/g, ' ')}
                   </div>
                   
                   <div
                    title="Permanently Delete Evidence"
                    onClick={(e) => {e.stopPropagation();handleDeleteCase(c.case_id);}}
                    style={{
                      background: 'rgba(239, 68, 68, 0.1)',
                      color: '#ef4444',
                      padding: '6px',
                      borderRadius: '6px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)'}>
                    
                     <Trash2 size={16} />
                   </div>

                   <div style={{ color: 'var(--muted)' }}>
                     {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                   </div>
                 </div>
              </div>

              {isExpanded &&
              <div style={{ marginTop: '24px', paddingTop: '24px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                  
                  {xaiEnabled && xaiLogs.length > 0 &&
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '16px', borderRadius: '12px', color: '#38bdf8', marginBottom: '24px' }}>
                        <div className="smallTitle" style={{ color: '#10b981', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Terminal size={16} /> Recovered AI Reasoning Logs
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                           {xaiLogs.map((log, idx) =>
                    <div key={idx} style={{ background: '#1e293b', padding: '12px', borderRadius: '8px', borderLeft: '3px solid #38bdf8' }}>
                                 <div className="mono" style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>PHASE: {log.stage}</div>
                                 {Object.entries(log.responses).filter(([_, val]) => val !== null).map(([agent, data]) =>
                      <div key={agent} style={{ marginTop: '8px' }}>
                                       <div style={{ color: '#e2e8f0', fontWeight: 600, textTransform: 'capitalize' }}>{agent}</div>
                                       {typeof data === 'string' ?
                        <div style={{ fontSize: '0.9rem', color: '#cbd5e1', marginTop: '4px', whiteSpace: 'pre-wrap' }}>{data}</div> :

                        <div style={{ fontSize: '0.9rem', color: '#cbd5e1', marginTop: '4px' }}>
                                             {data.diagnosis && <div><span style={{ color: '#94a3b8' }}>Diagnosis:</span> {data.diagnosis}</div>}
                                             {data.confidence !== undefined && <div><span style={{ color: '#94a3b8' }}>Confidence:</span> {data.confidence}%</div>}
                                             {(data.explanation || data.reasoning) && <div style={{ marginTop: '4px', whiteSpace: 'pre-wrap' }}><span style={{ color: '#94a3b8' }}>Reasoning:</span> {data.explanation || data.reasoning}</div>}
                                             {!data.diagnosis && !data.explanation && !data.reasoning && <div>{JSON.stringify(data)}</div>}
                                          </div>
                        }
                                    </div>
                      )}
                              </div>
                    )}
                        </div>
                     </div>
                }

                  <div style={{ marginBottom: '24px' }}>
                    <div className="smallTitle" style={{ marginBottom: '8px' }}>Full Clinical Complaint</div>
                    <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: 'var(--foreground)' }}>{c.user_message}</div>
                  </div>

                  {c.answered_followups && c.answered_followups.length > 0 &&
                <div style={{ marginBottom: '24px' }}>
                      <div className="smallTitle" style={{ marginBottom: '12px' }}>Patient Q&A Follow-ups</div>
                      <div style={{ display: 'grid', gap: '12px' }}>
                        {c.answered_followups.map((qa, i) =>
                    <div key={i} style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px' }}>
                             <div style={{ color: 'var(--accent-2)', marginBottom: '6px', fontWeight: 500 }}>Q: {qa.question}</div>
                             <div style={{ color: 'var(--foreground)' }}>A: {qa.answer}</div>
                           </div>
                    )}
                      </div>
                    </div>
                }

                  {c.consensus_winner && Object.keys(c.consensus_winner).length > 0 &&
                <div style={{ background: "rgba(34, 197, 94, 0.08)", border: "1px solid rgba(34, 197, 94, 0.3)", padding: '20px', borderRadius: '12px' }}>
                      <div className="smallTitle text-accent" style={{ marginBottom: '16px', fontSize: '1.25rem', fontWeight: 600 }}>Final Diagnostic Assessment</div>
                      
                      {c.consensus_winner.winner && c.consensus_winner.message &&
                  <div style={{ marginBottom: c.consensus_winner.diagnosis ? '20px' : '0' }}>
                           <h4 style={{ margin: '0 0 8px 0', color: 'var(--foreground)', fontSize: '1.1rem' }}>
                             Report from: {c.consensus_winner.winner}
                           </h4>
                           <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: 'var(--foreground)' }}>
                             <MarkdownText text={String(c.consensus_winner.message)} />
                           </div>
                        </div>
                  }

                      {c.consensus_winner.diagnosis && c.consensus_winner.explanation &&
                  <div style={{ display: 'grid', gap: '8px' }}>
                           <div><strong style={{ color: 'var(--foreground)' }}>Lead Specialist:</strong> {c.consensus_winner.winner}</div>
                           <div><strong style={{ color: 'var(--foreground)' }}>Final Diagnosis:</strong> {c.consensus_winner.diagnosis}</div>
                           <div style={{ marginTop: '8px', lineHeight: 1.5 }}>
                             <strong style={{ color: 'var(--foreground)' }}>Clinical Rationale:</strong><br />
                             <div style={{ color: 'var(--muted)', marginTop: '4px' }}>
                               <MarkdownText text={c.consensus_winner.explanation} />
                             </div>
                           </div>
                         </div>
                  }
                    </div>
                }

                </div>
              }
            </div>);

        })}
      </div>
    </div>);

}