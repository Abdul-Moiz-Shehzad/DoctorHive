import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, FileText, Activity, Play, Clock, ChevronRight } from 'lucide-react';
import { fetchAllCases } from '../api';
import { useAuth } from '../context/AuthContext';

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAllCases()
      .then(setCases)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const activeCases = cases.filter((c) => c.stage !== 'completed').length;
  const resolvedCases = cases.filter((c) => c.stage === 'completed').length;
  const recentCases = cases.slice(0, 5);

  const stageColor = (stage) =>
    stage === 'completed' ? 'var(--success-text)' : 'var(--accent-primary)';

  return (
    <div className="dashboard-page">
      {/* Header */}
      <header className="page-header">
        <div>
          <h1 className="title">Welcome back, {user?.username || 'User'} 👋</h1>
          <p className="subtitle">Here's an overview of your consultation activity.</p>
        </div>
        <button onClick={() => navigate('/consultation')} className="cta-button">
          <Play size={16} />
          New Consultation
        </button>
      </header>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{ color: 'var(--accent-primary)', background: 'var(--accent-glow)' }}>
            <Users size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '—' : activeCases}</span>
            <span className="stat-label">Active Cases</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ color: '#10b981', background: 'rgba(16,185,129,0.12)' }}>
            <FileText size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '—' : resolvedCases}</span>
            <span className="stat-label">Resolved Consults</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ color: '#f59e0b', background: 'rgba(245,158,11,0.12)' }}>
            <Activity size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '—' : cases.length}</span>
            <span className="stat-label">Total Intakes</span>
          </div>
        </div>
      </div>

      {/* Recent consultations */}
      <section className="card" style={{ marginTop: '24px' }}>
        <div className="cardTitle" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Clock size={16} style={{ color: 'var(--accent-primary)' }} />
            Recent Consultations
          </span>
          <button
            className="secondary"
            onClick={() => navigate('/history')}
            style={{ padding: '6px 14px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            View All <ChevronRight size={14} />
          </button>
        </div>

        {loading && (
          <div style={{ padding: '24px 0', color: 'var(--text-muted)', textAlign: 'center' }}>Loading cases…</div>
        )}

        {!loading && recentCases.length === 0 && (
          <div style={{ padding: '32px 0', color: 'var(--text-muted)', textAlign: 'center' }}>
            <Activity size={32} style={{ marginBottom: '8px', opacity: 0.4 }} />
            <div>No consultations yet. Start your first analysis!</div>
          </div>
        )}

        <div className="activity-list">
          {recentCases.map((c, i) => (
            <React.Fragment key={c.case_id}>
              <div className="activity-item">
                <div className="activity-info">
                  <span className="activity-id mono" style={{ fontSize: '0.82rem' }}>
                    {c.case_id.split('-')[0]}
                  </span>
                  <span className="activity-desc" style={{ maxWidth: '380px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {c.user_message || 'No message'}
                  </span>
                </div>
                <div className="chip">
                  <span style={{ color: stageColor(c.stage), textTransform: 'capitalize', fontSize: '12px', fontWeight: 600 }}>
                    {c.stage.replace(/_/g, ' ')}
                  </span>
                </div>
              </div>
              {i < recentCases.length - 1 && <div className="divider" style={{ margin: '10px 0' }} />}
            </React.Fragment>
          ))}
        </div>
      </section>
    </div>
  );
}