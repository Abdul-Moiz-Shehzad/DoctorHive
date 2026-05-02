import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Consultation from './pages/Consultation';
import History from './pages/History';
import Login from './pages/Login';
import Settings from './pages/Settings';
import PatientProfileModal from './components/PatientProfileModal';
import { AuthProvider, useAuth } from './context/AuthContext';
import { fetchProfile } from './api';

function PrivateRoute({ children }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function AppShell() {
  const { isAuthenticated, user } = useAuth();
  const [theme, setTheme] = useState(() => localStorage.getItem('doctorhive-theme') || 'dark');
  const [showProfileModal, setShowProfileModal] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('doctorhive-theme', theme);
  }, [theme]);

  // Check if logged-in user has a profile; show modal if not
  useEffect(() => {
    if (!isAuthenticated || !user) return;
    fetchProfile(user.user_id)
      .then((p) => { if (!p) setShowProfileModal(true); })
      .catch(() => setShowProfileModal(true));
  }, [isAuthenticated, user]);

  const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));

  if (!isAuthenticated) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  return (
    <div className="app-container">
      {showProfileModal && (
        <PatientProfileModal
          onClose={() => setShowProfileModal(false)}
          initialData={null}
          isSettings={false}
        />
      )}
      <Sidebar theme={theme} toggleTheme={toggleTheme} />
      <div className="main-content">
        <Routes>
          <Route path="/" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
          <Route path="/consultation" element={<PrivateRoute><Consultation /></PrivateRoute>} />
          <Route path="/history" element={<PrivateRoute><History /></PrivateRoute>} />
          <Route path="/settings" element={<PrivateRoute><Settings theme={theme} toggleTheme={toggleTheme} /></PrivateRoute>} />
          <Route path="/login" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </Router>
  );
}