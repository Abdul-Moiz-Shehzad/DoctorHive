import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { User, Shield, ShieldCheck, Moon, Sun, Bot, Sparkles } from 'lucide-react';
import { fetchProfile, postChangePassword, updatePreferredModel } from '../api';
import PatientProfileModal from '../components/PatientProfileModal';

export default function Settings({ theme, toggleTheme }) {
  const { user, updateUser } = useAuth();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  // Password state
  const [pwd, setPwd] = useState({ current: '', new: '', confirm: '' });
  const [pwdLoading, setPwdLoading] = useState(false);
  const [pwdError, setPwdError] = useState('');
  const [pwdSuccess, setPwdSuccess] = useState('');

  useEffect(() => {
    if (!user) return;
    fetchProfile(user.user_id)
      .then(setProfile)
      .catch(() => setProfile(null))
      .finally(() => setLoading(false));
  }, [user]);

  function handleModalClose(saved) {
    setShowModal(false);
    if (saved) {
      fetchProfile(user.user_id).then(setProfile).catch(() => {});
    }
  }

  async function handlePasswordChange(e) {
    e.preventDefault();
    setPwdError('');
    setPwdSuccess('');
    
    if (pwd.new !== pwd.confirm) {
      setPwdError('New passwords do not match');
      return;
    }
    
    setPwdLoading(true);
    try {
      await postChangePassword({ current_password: pwd.current, new_password: pwd.new });
      setPwdSuccess('Password updated successfully!');
      setPwd({ current: '', new: '', confirm: '' });
    } catch (err) {
      setPwdError(err.message || 'Failed to update password');
    } finally {
      setPwdLoading(false);
    }
  }

  async function handleModelChange(model) {
    try {
      await updatePreferredModel(model);
      updateUser({ preferred_model: model });
    } catch (err) {
      console.error("Failed to update model preference", err);
    }
  }

  const profileComplete = profile && (profile.age || profile.gender || profile.conditions?.length);

  return (
    <div className="page settings-page">
      {showModal && (
        <PatientProfileModal
          onClose={handleModalClose}
          initialData={profile}
          isSettings
        />
      )}

      <header className="page-header">
        <div>
          <h1 className="title">Settings & Profile</h1>
          <p className="subtitle">Manage your medical identity, security, and preferences.</p>
        </div>
      </header>

      <div className="settings-sections">
        {/* Medical profile section */}
        <section className="card">
          <div className="cardTitle" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Shield size={18} style={{ color: 'var(--accent-primary)' }} /> Medical Profile
            </span>
            <button className="cta-button" style={{ fontSize: '13px', padding: '8px 16px' }} onClick={() => setShowModal(true)}>
              {profileComplete ? 'Edit Profile' : 'Set Up Profile'}
            </button>
          </div>

          <div style={{ marginTop: '16px' }}>
            {loading ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Loading profile…</div>
            ) : !profileComplete ? (
              <div className="settings-empty" onClick={() => setShowModal(true)}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>🏥</div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>Incomplete Profile</div>
                <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Fill in your medical history to improve AI diagnostic accuracy.</div>
              </div>
            ) : (
              <div className="settings-profile-display">
                <div className="profile-grid-summary">
                   <div className="summary-item">
                     <span className="label">Age</span>
                     <span className="value">{profile.age}</span>
                   </div>
                   <div className="summary-item">
                     <span className="label">Gender</span>
                     <span className="value">{profile.gender}</span>
                   </div>
                   <div className="summary-item">
                     <span className="label">Blood Type</span>
                     <span className="value">{profile.blood_type || '—'}</span>
                   </div>
                </div>
                
                <div className="detail-list">
                  {profile.conditions?.length > 0 && (
                    <div className="detail-item">
                      <span className="label">Conditions</span>
                      <div className="chip-container">
                        {profile.conditions.map(c => <span key={c} className="chip">{c}</span>)}
                      </div>
                    </div>
                  )}
                  {profile.allergies?.length > 0 && (
                    <div className="detail-item">
                      <span className="label">Allergies</span>
                      <span className="value">{profile.allergies.join(', ')}</span>
                    </div>
                  )}
                  {profile.medications?.length > 0 && (
                    <div className="detail-item">
                      <span className="label">Medications</span>
                      <span className="value">{profile.medications.join(', ')}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>

        <div className="settings-grid-2col">
          {/* Account and Preferences */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Account info */}
            <section className="card">
              <div className="cardTitle"><User size={18} /> Account</div>
              <div className="settings-form" style={{ marginTop: '12px' }}>
                <div className="settings-info-row">
                  <span className="settings-label">Username</span>
                  <span className="settings-value">{user?.username}</span>
                </div>
                <div className="settings-info-row">
                  <span className="settings-label">Email</span>
                  <span className="settings-value">{user?.email}</span>
                </div>
              </div>
            </section>

            {/* Appearance Preferences */}
            <section className="card">
              <div className="cardTitle" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {theme === 'dark' ? <Moon size={18} /> : <Sun size={18} />} Appearance
                </span>
                <div className="theme-toggle-switch" onClick={toggleTheme}>
                  <div className={`switch-knob ${theme === 'dark' ? 'dark' : 'light'}`} />
                </div>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '12px 0 0 0' }}>
                Switch between dark and light mode for the best viewing experience.
              </p>
            </section>

            {/* AI Model Preferences */}
            <section className="card">
              <div className="cardTitle">
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Bot size={18} style={{ color: 'var(--accent-primary)' }} /> AI Intelligence
                </span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '8px 0 16px 0' }}>
                Choose the brain behind your consultation.
              </p>
              
              <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  className={`model-select-btn ${(user?.preferred_model || 'gemini') === 'gemini' ? 'active' : ''}`}
                  onClick={() => handleModelChange('gemini')}
                >
                  <Sparkles size={14} />
                  <span>Gemini 2.0</span>
                </button>
                <button 
                  className={`model-select-btn ${user?.preferred_model === 'gpt' ? 'active' : ''}`}
                  onClick={() => handleModelChange('gpt')}
                >
                  <Bot size={14} />
                  <span>GPT-4o</span>
                </button>
              </div>
            </section>
          </div>

          {/* Security / Password */}
          <section className="card">
            <div className="cardTitle"><ShieldCheck size={18} /> Security</div>
            <form onSubmit={handlePasswordChange} className="settings-form" style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className="profile-field">
                <label>Current Password</label>
                <input 
                  type="password" 
                  value={pwd.current} 
                  onChange={(e) => setPwd({ ...pwd, current: e.target.value })} 
                  placeholder="••••••••"
                  required
                />
              </div>
              <div className="profile-field">
                <label>New Password</label>
                <input 
                  type="password" 
                  value={pwd.new} 
                  onChange={(e) => setPwd({ ...pwd, new: e.target.value })} 
                  placeholder="••••••••"
                  required
                />
              </div>
              <div className="profile-field">
                <label>Confirm New Password</label>
                <input 
                  type="password" 
                  value={pwd.confirm} 
                  onChange={(e) => setPwd({ ...pwd, confirm: e.target.value })} 
                  placeholder="••••••••"
                  required
                />
              </div>
              
              {pwdError && <div className="login-error" style={{ fontSize: '12px' }}>{pwdError}</div>}
              {pwdSuccess && <div style={{ color: 'var(--success-text)', fontSize: '12px', background: 'var(--success-bg)', padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--success-border)' }}>{pwdSuccess}</div>}
              
              <button type="submit" className="cta-button" style={{ width: '100%', marginTop: '8px' }} disabled={pwdLoading}>
                {pwdLoading ? 'Updating…' : 'Update Password'}
              </button>
            </form>
          </section>
        </div>
      </div>
    </div>
  );
}
