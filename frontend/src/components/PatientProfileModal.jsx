import React, { useState } from 'react';
import { saveProfile } from '../api';
import { useAuth } from '../context/AuthContext';

const GENDERS = ['Male', 'Female', 'Non-binary', 'Prefer not to say'];
const BLOOD_TYPES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', "Don't know"];
const HABIT_OPTS = ['Never', 'Occasionally', 'Regularly'];
const COMMON_CONDITIONS = [
  'Diabetes Type 1', 'Diabetes Type 2', 'Hypertension', 'Asthma',
  'Heart Disease', 'Thyroid Disorder', 'Arthritis', 'Depression/Anxiety',
  'Kidney Disease', 'Liver Disease', 'Cancer (history)', 'Epilepsy'
];

export default function PatientProfileModal({ onClose, initialData = null, isSettings = false }) {
  const { user } = useAuth();
  const [step, setStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const [form, setForm] = useState({
    age: initialData?.age || '',
    gender: initialData?.gender || '',
    blood_type: initialData?.blood_type || '',
    allergies: initialData?.allergies?.join(', ') || '',
    conditions: initialData?.conditions || [],
    otherCondition: '',
    medications: initialData?.medications?.join(', ') || '',
    smoking: initialData?.smoking || 'Never',
    alcohol: initialData?.alcohol || 'Never',
    emergency_contact_name: initialData?.emergency_contact_name || '',
    emergency_contact_phone: initialData?.emergency_contact_phone || '',
  });

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }));

  function toggleCondition(c) {
    setForm(f => ({
      ...f,
      conditions: f.conditions.includes(c)
        ? f.conditions.filter(x => x !== c)
        : [...f.conditions, c]
    }));
  }

  function addOtherCondition() {
    const val = form.otherCondition.trim();
    if (val && !form.conditions.includes(val)) {
      setForm(f => ({ ...f, conditions: [...f.conditions, val], otherCondition: '' }));
    }
  }

  async function handleSave() {
    setSaving(true);
    setError('');
    try {
      await saveProfile({
        user_id: user.user_id,
        age: form.age ? parseInt(form.age) : null,
        gender: form.gender || null,
        blood_type: form.blood_type || null,
        allergies: form.allergies ? form.allergies.split(',').map(s => s.trim()).filter(Boolean) : [],
        conditions: form.conditions,
        medications: form.medications ? form.medications.split(',').map(s => s.trim()).filter(Boolean) : [],
        smoking: form.smoking || null,
        alcohol: form.alcohol || null,
        emergency_contact_name: form.emergency_contact_name || null,
        emergency_contact_phone: form.emergency_contact_phone || null,
      });
      onClose(true); // true = saved
    } catch (e) {
      setError(e.message || 'Failed to save');
    } finally {
      setSaving(false);
    }
  }

  const steps = [
    { title: 'Basic Information', icon: '👤' },
    { title: 'Medical History', icon: '🏥' },
    { title: 'Lifestyle & Emergency', icon: '💊' },
  ];

  return (
    <div className="profile-modal-overlay" onClick={isSettings ? undefined : (e) => e.target === e.currentTarget && onClose(false)}>
      <div className="profile-modal">
        {/* Header */}
        <div className="profile-modal-header">
          <div>
            <h2>{isSettings ? 'Edit Medical Profile' : 'Complete Your Medical Profile'}</h2>
            {!isSettings && <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '14px' }}>
              This helps our AI provide more accurate, personalized consultations.
            </p>}
          </div>
          {(isSettings || true) && (
            <button onClick={() => onClose(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '20px', padding: '4px' }}>✕</button>
          )}
        </div>

        {/* Step indicator */}
        <div className="profile-steps">
          {steps.map((s, i) => (
            <div key={i} className={`profile-step ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`} onClick={() => i < step && setStep(i)}>
              <div className="profile-step-dot">{i < step ? '✓' : i + 1}</div>
              <span>{s.title}</span>
            </div>
          ))}
        </div>

        {/* Step content */}
        <div className="profile-modal-body">
          {step === 0 && (
            <div className="profile-fields">
              <div className="profile-row">
                <div className="profile-field">
                  <label>Age</label>
                  <input type="number" min="1" max="120" placeholder="e.g. 28" value={form.age} onChange={set('age')} />
                </div>
                <div className="profile-field">
                  <label>Blood Type</label>
                  <select value={form.blood_type} onChange={set('blood_type')}>
                    <option value="">Select...</option>
                    {BLOOD_TYPES.map(b => <option key={b} value={b}>{b}</option>)}
                  </select>
                </div>
              </div>
              <div className="profile-field">
                <label>Gender</label>
                <div className="profile-chip-group">
                  {GENDERS.map(g => (
                    <button key={g} type="button" className={`profile-chip-btn ${form.gender === g ? 'selected' : ''}`} onClick={() => setForm(f => ({ ...f, gender: g }))}>{g}</button>
                  ))}
                </div>
              </div>
              <div className="profile-field">
                <label>Known Allergies <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(comma-separated)</span></label>
                <input type="text" placeholder="e.g. Penicillin, Pollen, Latex" value={form.allergies} onChange={set('allergies')} />
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="profile-fields">
              <div className="profile-field">
                <label>Pre-existing Conditions</label>
                <div className="profile-chip-group" style={{ flexWrap: 'wrap' }}>
                  {COMMON_CONDITIONS.map(c => (
                    <button key={c} type="button" className={`profile-chip-btn ${form.conditions.includes(c) ? 'selected' : ''}`} onClick={() => toggleCondition(c)}>{c}</button>
                  ))}
                </div>
                <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                  <input type="text" placeholder="Other condition..." value={form.otherCondition} onChange={set('otherCondition')} onKeyDown={e => e.key === 'Enter' && addOtherCondition()} style={{ flex: 1 }} />
                  <button type="button" onClick={addOtherCondition} className="secondary" style={{ padding: '8px 14px', fontSize: '13px' }}>Add</button>
                </div>
                {form.conditions.length > 0 && (
                  <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {form.conditions.map(c => (
                      <span key={c} className="chip" style={{ cursor: 'pointer', background: 'var(--accent-glow)' }} onClick={() => toggleCondition(c)}>{c} ✕</span>
                    ))}
                  </div>
                )}
              </div>
              <div className="profile-field">
                <label>Current Medications <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(comma-separated)</span></label>
                <input type="text" placeholder="e.g. Metformin 500mg, Lisinopril 10mg" value={form.medications} onChange={set('medications')} />
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="profile-fields">
              <div className="profile-row">
                <div className="profile-field">
                  <label>Smoking</label>
                  <div className="profile-chip-group">
                    {HABIT_OPTS.map(o => <button key={o} type="button" className={`profile-chip-btn ${form.smoking === o ? 'selected' : ''}`} onClick={() => setForm(f => ({ ...f, smoking: o }))}>{o}</button>)}
                  </div>
                </div>
                <div className="profile-field">
                  <label>Alcohol</label>
                  <div className="profile-chip-group">
                    {HABIT_OPTS.map(o => <button key={o} type="button" className={`profile-chip-btn ${form.alcohol === o ? 'selected' : ''}`} onClick={() => setForm(f => ({ ...f, alcohol: o }))}>{o}</button>)}
                  </div>
                </div>
              </div>
              <div className="profile-row">
                <div className="profile-field">
                  <label>Emergency Contact Name</label>
                  <input type="text" placeholder="Full name" value={form.emergency_contact_name} onChange={set('emergency_contact_name')} />
                </div>
                <div className="profile-field">
                  <label>Emergency Contact Phone</label>
                  <input type="tel" placeholder="+1 234 567 8900" value={form.emergency_contact_phone} onChange={set('emergency_contact_phone')} />
                </div>
              </div>
              {error && <div className="login-error">{error}</div>}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="profile-modal-footer">
          {!isSettings && step === 0 && (
            <button onClick={() => onClose(false)} className="secondary" style={{ fontSize: '13px' }}>Skip for now</button>
          )}
          {step > 0 && <button onClick={() => setStep(s => s - 1)} className="secondary">Back</button>}
          <div style={{ flex: 1 }} />
          {step < 2
            ? <button onClick={() => setStep(s => s + 1)} className="cta-button">Continue →</button>
            : <button onClick={handleSave} className="cta-button" disabled={saving}>{saving ? 'Saving…' : 'Save Profile'}</button>
          }
        </div>
      </div>
    </div>
  );
}
