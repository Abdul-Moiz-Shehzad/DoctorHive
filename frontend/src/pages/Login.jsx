import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Brain, Database, ShieldCheck, Lock, UserPlus, ArrowRight } from 'lucide-react';
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
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-panel-left">
        <Link to="/" className="login-brand" style={{ textDecoration: 'none' }}>
          <div className="login-logo">
            <img src="/doctorhive logo.png" alt="DoctorHive Logo" className="landing-brand-mark" style={{ borderRadius: '14px' }} />
            <span>DoctorHive</span>
          </div>
        </Link>

        <div className="login-hero-copy">
            <h1 className="login-headline">Clinical Intelligence<br />at your fingertips.</h1>
            <p className="login-subtext">
              Log in to manage complex consultations, review agent consensus, and access persistent clinical history in one unified workspace.
            </p>
          </div>

        <div className="login-feature-grid">
          {[
            { icon: <Brain size={24} />, label: 'Consensus Reasoning', desc: 'Specialist agents debate patient data to deliver a unified diagnosis.' },
            { icon: <Database size={24} />, label: 'Clinical Continuity', desc: 'Full case history and context preserved across every consultation.' },
            { icon: <ShieldCheck size={24} />, label: 'Enterprise Security', desc: 'Audit-ready encryption and secure medical data management.' },
          ].map((item) => (
            <div key={item.label} className="login-feature-block">
              <div className="login-feature-mark">{item.icon}</div>
              <div>
                <div className="login-feature-heading">{item.label}</div>
                <div className="login-feature-copy">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="login-panel-right">
        <div className="login-form-card glass-card">
          <div className="login-tabs">
            <button className={`login-tab ${tab === 'login' ? 'active' : ''}`} onClick={() => { setTab('login'); setError(''); }}>
              <Lock size={16} /> Sign In
            </button>
            <button className={`login-tab ${tab === 'register' ? 'active' : ''}`} onClick={() => { setTab('register'); setError(''); }}>
              <UserPlus size={16} /> Sign Up
            </button>
          </div>

          <div className="login-form-header">
            <h2>{tab === 'login' ? 'Welcome Back' : 'Get Started'}</h2>
            <p>{tab === 'login' ? 'Log in to continue your consultations.' : 'Start your first AI-powered medical review.'}</p>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            <div className="login-field">
              <label htmlFor="lf-email">Email Address</label>
              <input id="lf-email" type="email" placeholder="yourname@example.com" value={form.email} onChange={update('email')} required autoComplete="email" />
            </div>

            {tab === 'register' && (
              <div className="login-field">
                <label htmlFor="lf-username">Full Name / Username</label>
                <input id="lf-username" type="text" placeholder="e.g. John Doe" value={form.username} onChange={update('username')} required autoComplete="username" />
              </div>
            )}

            <div className="login-field">
              <label htmlFor="lf-password">Password</label>
              <input id="lf-password" type="password" placeholder="••••••••" value={form.password} onChange={update('password')} required autoComplete={tab === 'login' ? 'current-password' : 'new-password'} />
            </div>

            {error && <div className="login-error">{error}</div>}

            <button type="submit" className="login-submit" disabled={loading}>
              {loading ? <span className="login-spinner" /> : (
                <>
                  {tab === 'login' ? 'Log In' : 'Create Account'}
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </form>

          <p className="login-switch-text">
            {tab === 'login' ? 'New to DoctorHive?' : 'Already have an account?'}
            <button className="login-switch-btn" onClick={() => { setTab(tab === 'login' ? 'register' : 'login'); setError(''); }}>
              {tab === 'login' ? ' Create account' : ' Log in'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
