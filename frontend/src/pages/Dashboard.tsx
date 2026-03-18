import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, FileText, Activity, ArrowRight, Play } from 'lucide-react';

export default function Dashboard() {
  const navigate = useNavigate();

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
            <span className="stat-value">12</span>
            <span className="stat-label">Active Cases</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ color: 'var(--accent-2)', background: 'rgba(129, 140, 248, 0.1)' }}>
            <FileText size={24} />
          </div>
          <div className="stat-details">
            <span className="stat-value">48</span>
            <span className="stat-label">Resolved Consults</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ color: 'var(--success-text)', background: 'var(--success-bg)' }}>
            <Activity size={24} />
          </div>
          <div className="stat-details">
            <span className="stat-value">98%</span>
            <span className="stat-label">System Uptime</span>
          </div>
        </div>
      </div>

      <section className="recent-activity card mt-8">
        <div className="cardTitle">
          Recent Consultations
          <button className="secondary small-btn" style={{ marginLeft: 'auto', padding: '6px 12px', fontSize: '12px' }}>
            View All
          </button>
        </div>
        <div className="activity-list">
           <div className="activity-item">
             <div className="activity-info">
                <span className="activity-id mono">CASE-982</span>
                <span className="activity-desc">Cardiology Review</span>
             </div>
             <div className="chip">
                <span className="muted">Status</span>
                <span className="mono" style={{ color: 'var(--success-text)' }}>Resolved</span>
             </div>
           </div>
           
           <div className="divider" style={{ margin: '12px 0' }} />
           
           <div className="activity-item">
             <div className="activity-info">
                <span className="activity-id mono">CASE-981</span>
                <span className="activity-desc">Neurology Consult</span>
             </div>
             <div className="chip">
                <span className="muted">Status</span>
                <span className="mono" style={{ color: 'var(--accent-1)' }}>Pending</span>
             </div>
           </div>
        </div>
      </section>
    </div>
  );
}
