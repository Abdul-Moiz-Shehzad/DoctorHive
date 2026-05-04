import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  Brain,
  HeartPulse,
  Eye,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  Layers,
  ClipboardList,
  Users,
  CheckCircle2,
  Sun,
  Moon,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const featureCards = [
  {
    icon: <ShieldCheck size={22} />,
    title: 'Specialist consensus',
    description: 'Cardiology, neurology and ophthalmology agents debate together, then deliver a unified, clinician-ready diagnosis.',
    badge: 'Multi-agent review',
  },
  {
    icon: <ClipboardList size={22} />,
    title: 'Clinical continuity',
    description: 'Every case stores conversation history, patient context, and follow-up recommendations in a single secure record.',
    badge: 'Audit-ready memory',
  },
  {
    icon: <Eye size={22} />,
    title: 'Explainable output',
    description: 'Transparent reasoning helps care teams understand why a recommendation was made and what to do next.',
    badge: 'Clinician confidence',
  },
  {
    icon: <Sparkles size={22} />,
    title: 'Fast handoff',
    description: 'Move from intake to referral with an AI workflow built for reliable decision support and smoother clinical handoffs.',
    badge: 'Workflow-ready',
  },
];

const steps = [
  {
    icon: <Activity size={20} />,
    title: 'Collect case details',
    detail: 'Capture symptoms, history, and report data through an intuitive intake flow.',
  },
  {
    icon: <Brain size={20} />,
    title: 'Run specialist review',
    detail: 'Multiple expert agents cross-check the case and generate a reasoned consensus.',
  },
  {
    icon: <HeartPulse size={20} />,
    title: 'Deliver actionable care',
    detail: 'Share a concise diagnosis summary and follow-up plan with clinical teams instantly.',
  },
];

export default function Landing() {
  const { user, updateUser, isAuthenticated } = useAuth();
  const [theme, setTheme] = useState(() => {
    // Priority: User Preference (Cloud) > Local Storage > Document Attribute > Default Dark
    return user?.preferred_theme || localStorage.getItem('theme') || document.documentElement.getAttribute('data-theme') || 'dark';
  });

  // Sync theme when user logs in or preference changes in state
  useEffect(() => {
    if (user?.preferred_theme && user.preferred_theme !== theme) {
      setTheme(user.preferred_theme);
    }
  }, [user?.preferred_theme]);

  useEffect(() => {
    // Sync document attribute with state
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Persist to backend if logged in
    if (isAuthenticated) {
      import('../api').then(m => m.updatePreferredTheme(theme)).catch(console.error);
    }
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, [theme]);

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    localStorage.setItem('theme', next);
  };

  return (
    <div className="landing-page">
      <header className="landing-nav">
        <div className="landing-brand">
          <img src="/doctorhive logo.png" alt="DoctorHive Logo" className="landing-brand-mark" />
          <div>
            <div className="landing-brand-title">DoctorHive</div>
            <div className="landing-brand-subtitle">Multi-Agent Medical Debate System</div>
          </div>
        </div>

        <div className="landing-nav-actions">
          <button className="theme-toggle" onClick={toggleTheme} title="Toggle Theme">
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <Link to={isAuthenticated ? "/dashboard" : "/login"} className="landing-nav-link">
            {isAuthenticated ? 'Dashboard' : 'Login'}
          </Link>
          <Link to={isAuthenticated ? "/dashboard" : "/login"} className="landing-nav-button">Get started</Link>
        </div>
      </header>

      <main className="landing-hero">
        <div className="landing-copy reveal reveal-left">
          <span className="landing-eyebrow">Clinical AI for trusted care teams</span>
          <h1>Bring specialist debate, case continuity, and diagnosis confidence into one modern workflow.</h1>
          <p>DoctorHive helps clinicians review patient intake, agent feedback, and longitudinal case history with clarity and control.</p>

          <div className="landing-cta-group">
            <Link to={isAuthenticated ? "/dashboard" : "/login"} className="landing-cta-button">Start your first case</Link>
            <a href="#expert-flow" className="landing-cta-secondary">See the workflow</a>
          </div>

          <div className="landing-stat-strip">
            <div className="reveal reveal-up reveal-delay-1">
              <strong>Multi-Agent</strong>
              <span>Reasoning Engine</span>
            </div>
            <div className="reveal reveal-up reveal-delay-2">
              <strong>Explainable</strong>
              <span>AI Diagnostics</span>
            </div>
            <div className="reveal reveal-up reveal-delay-3">
              <strong>Transparent</strong>
              <span>Clinical Trials</span>
            </div>
          </div>
        </div>

        <div className="landing-preview-panel reveal reveal-right">
          <div className="hero-card hero-card-main full-width-hero">
            <div className="hero-card-badge">Interactive Consultation Workspace</div>
            <h2>Full Case Transparency</h2>
            <p>From initial intake to final specialist consensus—track every reasoning step in a unified, audit-ready clinical environment.</p>
            
            <div className="agent-focus-strip">
              <span className="focus-label">Specialist Agents:</span>
              <div className="hero-card-tags">
                <span><HeartPulse size={14} /> Cardiology</span>
                <span><Brain size={14} /> Neurology</span>
                <span><Eye size={14} /> Ophthalmology</span>
              </div>
            </div>

            <div className="hero-metrics">
              <div className="reveal reveal-up reveal-delay-1">
                <strong>Audit-Ready</strong>
                <span>Clinical Continuity</span>
              </div>
              <div className="reveal reveal-up reveal-delay-2">
                <strong>Reasoning</strong>
                <span>Step-by-Step Tracing</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      <section id="expert-flow" className="landing-flow">
        <div className="landing-section-header flow-header reveal">
          <p className="landing-section-eyebrow">Interaction hub</p>
          <h2>Patient intake routes through intelligent specialist debate.</h2>
        </div>

        <div className="landing-flow-vertical">
          <div className="flow-step reveal reveal-left">
            <div className="flow-step-icon"><Users size={24} /></div>
            <div className="flow-step-content">
              <strong>Patient intake</strong>
              <span>Symptoms, history and scanned reports collected via intuitive UI.</span>
            </div>
          </div>
          
          <div className="flow-connector"></div>

          <div className="flow-step reveal reveal-left">
            <div className="flow-step-icon"><Activity size={24} /></div>
            <div className="flow-step-content">
              <strong>Orchestration</strong>
              <span>General Practice Agent routes the case to relevant specialists.</span>
            </div>
          </div>

          <div className="flow-connector"></div>

          <div className="flow-step reveal reveal-left">
            <div className="flow-step-icon"><Brain size={24} /></div>
            <div className="flow-step-content">
              <strong>Specialist Debate</strong>
              <div className="specialist-logos">
                <span><HeartPulse size={16} /> Cardiology</span>
                <span><Brain size={16} /> Neurology</span>
                <span><Eye size={16} /> Ophthalmology</span>
              </div>
              <span>Expert agents cross-reference data to reach a consensus.</span>
            </div>
          </div>

          <div className="flow-connector"></div>

          <div className="flow-step reveal reveal-left">
            <div className="flow-step-icon"><CheckCircle2 size={24} /></div>
            <div className="flow-step-content">
              <strong>Refined Diagnosis</strong>
              <span>A reasoned, clinician-ready summary with actionable next steps.</span>
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="landing-features">
        <div className="landing-section-header reveal">
          <p className="landing-section-eyebrow">Why teams choose DoctorHive</p>
          <h2>High-value capabilities for modern healthcare workflows.</h2>
        </div>

        <div className="landing-feature-grid">
          {featureCards.map((feature, idx) => (
            <div key={feature.title} className={`landing-feature-card reveal reveal-up reveal-delay-${idx + 1}`}>
              <div className="landing-feature-icon">{feature.icon}</div>
              <div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
                <span>{feature.badge}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="landing-steps">
        <div className="landing-section-header reveal">
          <p className="landing-section-eyebrow">Process made simple</p>
          <h2>From intake to recommendation in three smooth stages.</h2>
        </div>

        <div className="landing-step-grid">
          {steps.map((item, idx) => (
            <div key={item.title} className={`landing-step-card reveal reveal-up reveal-delay-${idx + 1}`}>
              <div className="landing-step-number">{item.icon}</div>
              <div>
                <h4>{item.title}</h4>
                <p>{item.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <Link to="/" className="landing-brand" style={{ textDecoration: 'none' }}>
              <img src="/doctorhive logo.png" alt="DoctorHive Logo" className="landing-brand-mark" />
              <div className="landing-brand-title">DoctorHive</div>
            </Link>
            <p>Advancing clinical decision support through multi-agent AI debate and transparent reasoning.</p>
          </div>

          <div className="footer-actions">
            <Link to={isAuthenticated ? "/dashboard" : "/login"} className="landing-cta-button">Start your first case</Link>
            <div className="footer-bottom-text">
              &copy; 2026 DoctorHive AI. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
