import React, { useEffect, useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { LayoutDashboard, MessageSquareText, History, Settings, ActivitySquare, Moon, Sun, LogOut, ChevronRight, Clock } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { fetchAllCases } from '../api';

export default function Sidebar({ theme, toggleTheme }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [recentCases, setRecentCases] = useState([]);

  useEffect(() => {
    fetchAllCases()
      .then(data => setRecentCases(data.slice(0, 3)))
      .catch(() => {});
  }, []);

  function handleLogout() {
    logout();
    navigate('/login', { replace: true });
  }

  const initials = user?.username ? user.username.slice(0, 2).toUpperCase() : 'DR';

  const stageLabel = (stage) => stage?.replace(/_/g, ' ') || '';
  const isCompleted = (stage) => stage === 'completed';

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <ActivitySquare className="logo-icon" size={28} />
        <h2>DoctorHive</h2>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'} end>
          <LayoutDashboard size={20} />
          <span>Dashboard</span>
        </NavLink>

        <NavLink to="/consultation" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          <MessageSquareText size={20} />
          <span>Consultation</span>
        </NavLink>

        {/* Chat History nav with inline recent chats */}
        <div className="nav-group">
          <NavLink to="/history" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            <History size={20} />
            <span>Chat History</span>
          </NavLink>

          {recentCases.length > 0 && (
            <div className="sidebar-recent-chats">
              {recentCases.map(c => (
                <button
                  key={c.case_id}
                  className="sidebar-recent-item"
                  onClick={() => navigate('/consultation', { state: { resumeCaseId: c.case_id } })}
                  title={c.user_message}
                >
                  <div className="sidebar-recent-dot" style={{ background: isCompleted(c.stage) ? 'var(--success-text)' : 'var(--accent-primary)' }} />
                  <span className="sidebar-recent-text">
                    {c.user_message?.length > 28 ? c.user_message.substring(0, 28) + '…' : (c.user_message || 'No message')}
                  </span>
                  <span className="sidebar-recent-stage">{isCompleted(c.stage) ? '✓' : '...'}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </nav>

      <div className="sidebar-footer">
        {user && (
          <div className="sidebar-user-card">
            <div className="sidebar-avatar">{initials}</div>
            <div className="sidebar-user-info">
              <span className="sidebar-username">{user.username}</span>
              <span className="sidebar-email">{user.email}</span>
            </div>
          </div>
        )}

        <NavLink to="/settings" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          <Settings size={18} />
          <span>Settings</span>
        </NavLink>

        <button
          onClick={handleLogout}
          className="nav-link logout-btn"
          style={{ width: 'auto', background: 'transparent', border: 'none', cursor: 'pointer' }}
        >
          <LogOut size={18} />
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  );
}