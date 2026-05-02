import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, FileText, Activity, ArrowRight, Play } from 'lucide-react';
import { fetchAllCases, type CaseHistory } from '../api';

export default function Dashboard() {
  const navigate = useNavigate();
  const [cases, setCases] = useState<CaseHistory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAllCases()
      .then(setCases)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const activeCases = cases.filter(c => c.stage !== 'completed').length;
  const resolvedCases = cases.filter(c => c.stage === 'completed').length;
  // Grab top 5 most recent
  const recentCases = cases.slice(0, 5); 

  return (
    <div className="dashboard-page">
      <header className="page-header">
        <div>
          <h1 className="title">Welcome back, Dr. User</h1>
          <p className="subtitle">Here's what's happening today.</p>
        </div>
        <button onClick={() => navigate('/consultation')} className="cta-button">
          <Play size={18} />
          New Consultation
        </button>
      </header>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{ color: 'var(--accent-1)', background: 'rgba(168, 85, 247, 0.1)' }}>
            <Users size={24} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? "-" : activeCases}</span>
            <span className="stat-label">Active Cases</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ color: 'var(--accent-2)', background: 'rgba(129, 140, 248, 0.1)' }}>
            <FileText size={24} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? "-" : resolvedCases}</span>
            <span className="stat-label">Resolved Consults</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ color: 'var(--success-text)', background: 'var(--success-bg)' }}>
            <Activity size={24} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? "-" : cases.length}</span>
            <span className="stat-label">Total Intakes</span>
          </div>
        </div>
      </div>

      <section className="recent-activity card mt-8">
        <div className="cardTitle">
          Recent Consultations
          <button className="secondary small-btn" onClick={() => navigate('/history')} style={{ marginLeft: 'auto', padding: '6px 12px', fontSize: '12px' }}>
            View Full History
          </button>
        </div>
        
        {loading && <div style={{ color: 'var(--muted)', marginTop: '16px' }}>Loading cases...</div>}
        
        {!loading && recentCases.length === 0 && (
           <div style={{ color: 'var(--muted)', marginTop: '16px' }}>No consultations found. Be the first to start an analysis!</div>
        )}

        <div className="activity-list">
           {recentCases.map((c, i) => (
             <React.Fragment key={c.case_id}>
               <div className="activity-item">
                 <div className="activity-info">
                    <span className="activity-id mono" style={{ fontSize: '0.85rem' }}>{c.case_id.split('-')[0]}</span>
                    <span className="activity-desc" style={{ maxWidth: '400px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {c.user_message}
                    </span>
                 </div>
                 <div className="chip">
                    <span className="muted">Stage</span>
                    <span className="mono" style={{ textTransform: 'capitalize', color: c.stage === 'completed' ? 'var(--success-text)' : 'var(--accent-1)' }}>
                      {c.stage.replace(/_/g, ' ')}
                    </span>
                 </div>
               </div>
               
               {i < recentCases.length - 1 && <div className="divider" style={{ margin: '12px 0' }} />}
             </React.Fragment>
           ))}
        </div>
      </section>
    </div>
  );
}
