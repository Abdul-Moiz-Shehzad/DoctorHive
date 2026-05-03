import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { fetchAllCases, deleteCase } from '../api';
import { Clock, Users, ChevronDown, ChevronUp, Trash2, Terminal, Play, Lock, MessageSquare } from 'lucide-react';

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

export default function History() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    if (!user) return;
    async function load() {
      try {
        const data = await fetchAllCases(user.user_id);
        setCases(data);
      } catch (e) {
        // If it's a 404, we treat it as no cases
        if (e.message?.includes('404')) {
          setCases([]);
        } else {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [user]);

  const handleDeleteCase = async (caseId) => {
    const previousCases = [...cases];
    setCases((c) => c.filter((x) => x.case_id !== caseId));
    try {
      await deleteCase(caseId);
    } catch (e) {
      console.error('Failed to delete case:', e);
      setCases(previousCases);
      setError('Deletion failed. Try refreshing.');
    }
  };

  const handleContinue = (caseId) => {
    navigate('/consultation', { state: { resumeCaseId: caseId } });
  };

  return (
    <div className="page history-page">
      <header className="page-header">
        <div>
          <h1 className="title">Chat History</h1>
          <p className="subtitle">All your past and ongoing AI consultations.</p>
        </div>
      </header>

      {loading && (
        <div className="status-container" style={{ textAlign: 'center', padding: '60px 0' }}>
          <Clock className="spin" size={40} style={{ color: 'var(--accent-primary)', marginBottom: '16px' }} />
          <div style={{ color: 'var(--text-muted)' }}>Loading your history…</div>
        </div>
      )}

      {error && (
        <div className="error-card" style={{ margin: '24px 0', padding: '20px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '12px', color: '#ef4444' }}>
          <strong>Error loading history:</strong> {error}
        </div>
      )}

      {!loading && !error && cases.length === 0 && (
        <div className="empty-state-card" style={{ textAlign: 'center', padding: '80px 20px', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '20px', margin: '24px 0' }}>
          <MessageSquare size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px', opacity: 0.3 }} />
          <h3 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>No Previous Chats</h3>
          <p style={{ color: 'var(--text-muted)', maxWidth: '400px', margin: '0 auto 24px auto' }}>You haven't started any consultations yet. Launch a new analysis from the dashboard.</p>
          <button className="cta-button" onClick={() => navigate('/consultation')}>Start Consultation</button>
        </div>
      )}

      <div className="cases-list" style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '24px' }}>
        {cases.map((c) => {
          const isExpanded = expandedId === c.case_id;
          const closed = c.stage === 'completed';

          let xaiLogs = [];
          if (isExpanded) {
            const raw = localStorage.getItem(`xai_${c.case_id}`);
            xaiLogs = raw ? JSON.parse(raw) : [];
          }

          const badgeColor = closed ? 'var(--success-text)' : 'var(--accent-primary)';

          return (
            <div key={c.case_id} className="card history-card" style={{ cursor: 'pointer' }} onClick={() => setExpandedId(isExpanded ? null : c.case_id)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="mono" style={{ fontSize: '0.82rem', color: 'var(--accent-primary)', marginBottom: '6px' }}>{c.case_id}</div>
                  <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '500px' }}>
                    {c.chat_name || (c.user_message?.length > 80 ? c.user_message.substring(0, 80) + '…' : c.user_message)}
                  </div>
                  <div style={{ display: 'flex', gap: '16px', marginTop: '10px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                      <Clock size={13} /> {c.timestamp ? new Date(c.timestamp).toLocaleString() : 'Unknown'}
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                      <Users size={13} /> {c.specialists_required?.length || 0} Specialists
                    </span>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0, marginLeft: '16px' }}>
                  {/* Status badge */}
                  <div className="chip" style={{ color: badgeColor, border: `1px solid ${badgeColor}40`, textTransform: 'capitalize', fontSize: '12px' }}>
                    {closed ? '✓ Closed' : c.stage.replace(/_/g, ' ')}
                  </div>

                  {/* Continue / Closed action */}
                  {closed ? (
                    <div title="Case closed — view only" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: 'var(--text-muted)', padding: '5px 10px', border: '1px solid var(--border-color)', borderRadius: '6px' }}>
                      <Lock size={13} /> Closed
                    </div>
                  ) : (
                    <button
                      className="cta-button"
                      style={{ padding: '6px 14px', fontSize: '12px', gap: '4px' }}
                      onClick={(e) => { e.stopPropagation(); handleContinue(c.case_id); }}
                    >
                      <Play size={13} /> Continue
                    </button>
                  )}

                  {/* Delete */}
                  <div
                    title="Delete"
                    onClick={(e) => { e.stopPropagation(); handleDeleteCase(c.case_id); }}
                    style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', padding: '6px', borderRadius: '6px', display: 'flex', cursor: 'pointer', transition: 'all 0.2s' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(239,68,68,0.2)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'rgba(239,68,68,0.1)'}
                  >
                    <Trash2 size={15} />
                  </div>

                  <div style={{ color: 'var(--text-muted)' }}>
                    {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                  </div>
                </div>
              </div>

              {isExpanded && (
                <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px solid var(--border-color)' }}>
                  {xaiLogs.length > 0 && (
                    <div style={{ background: '#0f172a', border: '1px solid #1e293b', padding: '16px', borderRadius: '12px', color: '#38bdf8', marginBottom: '20px' }}>
                      <div style={{ color: '#10b981', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 600 }}>
                        <Terminal size={14} /> Recovered AI Reasoning Logs
                      </div>
                      {xaiLogs.map((log, idx) => (
                        <div key={idx} style={{ background: '#1e293b', padding: '12px', borderRadius: '8px', borderLeft: '3px solid #38bdf8', marginBottom: '8px' }}>
                          <div className="mono" style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Phase: {log.stage}</div>
                          {Object.entries(log.responses).filter(([_, v]) => v !== null).map(([agent, data]) => (
                            <div key={agent} style={{ marginTop: '8px' }}>
                              <div style={{ color: '#e2e8f0', fontWeight: 600, textTransform: 'capitalize', fontSize: '13px' }}>{agent}</div>
                              <div style={{ fontSize: '12px', color: '#cbd5e1', marginTop: '4px' }}>
                                {data.diagnosis && <div><span style={{ color: '#94a3b8' }}>Diagnosis:</span> {data.diagnosis} {data.confidence ? `(${data.confidence}%)` : ''}</div>}
                                {data.explanation && <div style={{ marginTop: '4px', whiteSpace: 'pre-wrap' }}><span style={{ color: '#94a3b8' }}>Rationale:</span> {data.explanation}</div>}
                              </div>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}

                  <div style={{ marginBottom: '20px' }}>
                    <div className="smallTitle" style={{ marginBottom: '8px' }}>Clinical Complaint</div>
                    <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: 'var(--text-primary)' }}>{c.user_message}</div>
                  </div>

                  {c.snapshot?.gp_response && (
                    <div style={{ marginBottom: '20px' }}>
                      <div className="smallTitle" style={{ marginBottom: '8px' }}>GP Assessment</div>
                      <div className="chat-bubble ai ai-box" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '16px' }}>
                        <MarkdownText text={c.snapshot.gp_response} />
                      </div>
                    </div>
                  )}

                  {c.answered_followups?.length > 0 && (
                    <div style={{ marginBottom: '20px' }}>
                      <div className="smallTitle" style={{ marginBottom: '10px' }}>Patient Q&A</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {c.answered_followups.map((qa, i) => (
                          <div key={i} style={{ background: 'var(--bg-tertiary)', padding: '12px', border: '1px solid var(--border-color)', borderRadius: '8px' }}>
                            <div style={{ color: 'var(--accent-secondary)', marginBottom: '4px', fontWeight: 500, fontSize: '13px' }}>Q: {qa.question}</div>
                            <div style={{ color: 'var(--text-primary)', fontSize: '13px' }}>A: {qa.answer}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {c.consensus_winner && Object.keys(c.consensus_winner).length > 0 && (
                    <div style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.3)', padding: '20px', borderRadius: '12px' }}>
                      <div className="smallTitle text-accent" style={{ marginBottom: '12px', fontSize: '1.1rem' }}>Final Diagnostic Assessment</div>
                      {c.consensus_winner.winner && c.consensus_winner.message && (
                        <div style={{ marginBottom: '16px' }}>
                          <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-primary)' }}>Report from: {c.consensus_winner.winner}</h4>
                          <MarkdownText text={String(c.consensus_winner.message)} />
                        </div>
                      )}
                      {c.consensus_winner.diagnosis && (
                        <div style={{ display: 'grid', gap: '6px', fontSize: '14px' }}>
                          <div><strong>Lead Specialist:</strong> {c.consensus_winner.winner}</div>
                          <div><strong>Final Diagnosis:</strong> {c.consensus_winner.diagnosis}</div>
                          {c.consensus_winner.explanation && (
                            <div style={{ marginTop: '6px' }}>
                              <strong>Rationale:</strong>
                              <div style={{ color: 'var(--text-muted)', marginTop: '4px', lineHeight: 1.5 }}>
                                <MarkdownText text={c.consensus_winner.explanation} />
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}