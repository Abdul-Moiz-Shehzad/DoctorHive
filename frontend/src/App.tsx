import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Consultation from './pages/Consultation';

export default function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <div className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/consultation" element={<Consultation />} />
            <Route path="/history" element={<div className="page"><h1 className="title">Patient History</h1><p className="subtitle">Coming soon...</p></div>} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}
