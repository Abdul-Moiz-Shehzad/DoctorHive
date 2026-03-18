import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, MessageSquareText, Activity, Settings, ActivitySquare } from 'lucide-react';

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <ActivitySquare className="logo-icon" size={28} />
        <h2>DoctorHive</h2>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')} end>
          <LayoutDashboard size={20} />
          <span>Dashboard</span>
        </NavLink>

        <NavLink to="/consultation" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
          <MessageSquareText size={20} />
          <span>Consultation</span>
        </NavLink>

        <NavLink to="/history" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
          <Activity size={20} />
          <span>Patient History</span>
        </NavLink>
      </nav>

      <div className="sidebar-footer">
        <button className="nav-link secondary" style={{ width: '100%', justifyContent: 'flex-start', background: 'transparent' }}>
          <Settings size={20} />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  );
}
