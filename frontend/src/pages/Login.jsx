import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { postLogin, postRegister } from '../api';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [tab, setTab] = useState('login'); // 'login' | 'register'
  const [form, setForm] = useState({ username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const update = (k) => (e) => setForm((prev) => ({ ...prev, [k]: e.target.value }));

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      let res;
      if (tab === 'login') {
        res = await postLogin({ email: form.email, password: form.password });
      } else {
        if (!form.email) { setError('Email is required'); setLoading(false); return; }
        res = await postRegister({ username: form.username, email: form.email, password: form.password });
      }
      login(res.access_token, { user_id: res.user_id, username: res.username, email: res.email });
      navigate('/');
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      {/* ── Left panel ── */}
      <div className="login-panel-left">
        <div className="login-brand">
          <div className="login-logo">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <rect width="40" height="40" rx="12" fill="url(#g1)" />
              <path d="M20 8v24M8 20h24" stroke="#fff" strokeWidth="3.5" strokeLinecap="round"/>
              <defs>
                <linearGradient id="g1" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#6366f1"/>
                  <stop offset="1" stopColor="#8b5cf6"/>
                </linearGradient>
              </defs>
            </svg>
            <span>DoctorHive</span>
          </div>
          <h1 className="login-headline">AI-Powered<br />Medical Intelligence</h1>
          <p className="login-subtext">
            A multi-agent diagnostic platform where specialist AI models collaborate in real-time to deliver accurate, explainable medical assessments.
          </p>
        </div>

        <div className="login-features">
          {[
            { icon: '🧠', label: 'Multi-Agent Consensus', desc: 'Neurologist, Cardiologist & more debate your case' },
            { icon: '🔍', label: 'Explainable AI', desc: 'Full transparency into every diagnostic decision' },
            { icon: '📋', label: 'Session Memory', desc: 'Full consultation history saved per patient' },
          ].map((f) => (
            <div key={f.label} className="login-feature-item">
              <span className="login-feature-icon">{f.icon}</span>
              <div>
                <div className="login-feature-label">{f.label}</div>
                <div className="login-feature-desc">{f.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Right panel (form) ── */}
      <div className="login-panel-right">
        <div className="login-form-card">
          <div className="login-tabs">
            <button className={`login-tab ${tab === 'login' ? 'active' : ''}`} onClick={() => { setTab('login'); setError(''); }}>
              Sign In
            </button>
            <button className={`login-tab ${tab === 'register' ? 'active' : ''}`} onClick={() => { setTab('register'); setError(''); }}>
              Create Account
            </button>
          </div>

          <div className="login-form-header">
            <h2>{tab === 'login' ? 'Welcome back' : 'Get started'}</h2>
            <p>{tab === 'login' ? 'Sign in to your DoctorHive account' : 'Create your DoctorHive account'}</p>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            <div className="login-field">
              <label htmlFor="lf-email">Email</label>
              <input id="lf-email" type="email" placeholder="Enter your email" value={form.email} onChange={update('email')} required autoComplete="email" />
            </div>

            {tab === 'register' && (
              <div className="login-field">
                <label htmlFor="lf-username">Username</label>
                <input id="lf-username" type="text" placeholder="Choose a username" value={form.username} onChange={update('username')} required autoComplete="username" />
              </div>
            )}

            <div className="login-field">
              <label htmlFor="lf-password">Password</label>
              <input id="lf-password" type="password" placeholder="Enter your password" value={form.password} onChange={update('password')} required autoComplete={tab === 'login' ? 'current-password' : 'new-password'} />
            </div>

            {error && <div className="login-error">{error}</div>}

            <button type="submit" className="login-submit" disabled={loading}>
              {loading ? (
                <span className="login-spinner" />
              ) : (
                tab === 'login' ? 'Sign In' : 'Create Account'
              )}
            </button>
          </form>

          <p className="login-switch-text">
            {tab === 'login' ? "Don't have an account?" : 'Already have an account?'}
            <button className="login-switch-btn" onClick={() => { setTab(tab === 'login' ? 'register' : 'login'); setError(''); }}>
              {tab === 'login' ? ' Sign Up' : ' Sign In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
