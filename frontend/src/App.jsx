import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8000/api';
const UPLOADS_BASE = 'http://localhost:8000/uploads';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [reportsHistory, setReportsHistory] = useState([]);
  
  // Form State
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('Male');
  const [symptoms, setSymptoms] = useState('');
  const [history, setHistory] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  
  // Loading & Processing state
  const [loading, setLoading] = useState(false);
  const [currentCase, setCurrentCase] = useState(null);
  
  // RAG Chat Drawer state
  const [chatQuery, setChatQuery] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  // Phase 2: 3-state image viewer — 'original' | 'mask' | 'heatmap'
  const [imageView, setImageView] = useState('mask');
  // Phase 2: History search filter
  const [historySearch, setHistorySearch] = useState('');
  const [severityFilter, setSeverityFilter] = useState('all');
  // Keep showMask alias for backwards compat
  const showMask = imageView === 'mask';
  
  // Interactive UI effects state
  const [scrollPercent, setScrollPercent] = useState(0);
  const [glowPos, setGlowPos] = useState({ x: -1000, y: -1000 });
  const targetGlowPos = useRef({ x: -1000, y: -1000 });
  
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const dashboardRef = useRef(null);
  const resultsRef = useRef(null);

  // Fetch report history on load
  useEffect(() => {
    fetchHistory();
  }, []);

  // Scroll RAG chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Track scroll progress
  useEffect(() => {
    const handleScroll = () => {
      const wins = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      setScrollPercent(docHeight > 0 ? (wins / docHeight) * 100 : 0);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Track mouse coordinates for background spotlight glow
  useEffect(() => {
    const handleMouseMove = (e) => {
      targetGlowPos.current = { x: e.clientX, y: e.clientY };
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Smooth mouse follower loop using Linear Interpolation (lerp)
  useEffect(() => {
    let animId;
    const updateGlow = () => {
      setGlowPos(prev => {
        // Safe check for startup coordinates
        if (prev.x === -1000) return targetGlowPos.current;
        
        const dx = targetGlowPos.current.x - prev.x;
        const dy = targetGlowPos.current.y - prev.y;
        
        // 8% step speed for organic lag-free fluid easing
        return {
          x: prev.x + dx * 0.08,
          y: prev.y + dy * 0.08
        };
      });
      animId = requestAnimationFrame(updateGlow);
    };
    updateGlow();
    return () => cancelAnimationFrame(animId);
  }, []);

  // Intersection Observer for scroll triggered reveal effects
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
        }
      });
    }, { threshold: 0.1 });
    
    const elements = document.querySelectorAll('.reveal-on-scroll');
    elements.forEach(el => observer.observe(el));
    
    return () => {
      elements.forEach(el => observer.unobserve(el));
      observer.disconnect();
    };
  }, [activeTab, currentCase, reportsHistory]);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/reports/`);
      if (res.ok) {
        const data = await res.json();
        setReportsHistory(data);
      }
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const scrollToDashboard = () => {
    dashboardRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please select or upload a medical image.');
      return;
    }
    
    setLoading(true);
    setCurrentCase(null);
    setChatMessages([]);
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('patient_name', name);
    formData.append('patient_age', age);
    formData.append('patient_sex', sex);
    formData.append('symptoms', symptoms);
    formData.append('medical_history', history);

    try {
      const res = await fetch(`${API_BASE}/analysis/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (res.ok) {
        const data = await res.json();
        setCurrentCase(data);
        fetchHistory(); // refresh history
        
        // Seed first RAG guidance prompt
        if (data.id) {
          triggerInitialRAG(data);
        }

        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
      } else {
        const err = await res.json();
        alert(`Analysis Error: ${err.detail || 'Failed to analyze case'}`);
      }
    } catch (err) {
      console.error('Upload error:', err);
      alert('Could not reach the analysis backend. Check if FastAPI is running.');
    } finally {
      setLoading(false);
    }
  };

  const triggerInitialRAG = async (report) => {
    setChatLoading(true);
    const initialQuery = `Provide initial recommendations and suggestions based on ${report.disease}`;
    setChatMessages([{ role: 'user', content: initialQuery }]);
    
    try {
      const res = await fetch(`${API_BASE}/chat/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          report_id: report.id,
          query: initialQuery,
          history: []
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setChatMessages(prev => [
          ...prev,
          { role: 'assistant', content: data.answer }
        ]);
        setCurrentCase(prev => ({
          ...prev,
          rag_suggestions: data.suggestions
        }));
      }
    } catch (err) {
      console.error('Initial RAG loading failed:', err);
    } finally {
      setChatLoading(false);
    }
  };

  const handleChatSend = async (e) => {
    e.preventDefault();
    if (!chatQuery.trim() || !currentCase) return;
    
    const userMsg = chatQuery;
    setChatQuery('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          report_id: currentCase.id,
          query: userMsg,
          history: chatMessages
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setChatMessages(prev => [
          ...prev,
          { role: 'assistant', content: data.answer }
        ]);
      }
    } catch (err) {
      console.error('Chat query error:', err);
      setChatMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Connection timed out. Could not fetch guidance advice.' }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const selectHistoricCase = async (report) => {
    setCurrentCase(report);
    setChatMessages([]);
    setActiveTab('dashboard');
    setTimeout(() => {
      dashboardRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    
    if (report.rag_suggestions && report.rag_suggestions.length > 0) {
      setChatMessages([
        { role: 'user', content: 'Load case report clinical advice history.' },
        { role: 'assistant', content: `Here is the clinical review checklist generated for ${report.patient_name}:\n` + 
          report.rag_suggestions.map(s => `- [ ] ${s}`).join('\n') }
      ]);
    }
  };

  const getBBoxStyle = (bbox) => {
    if (!bbox || bbox.length !== 4) return { display: 'none' };
    const [xmin, ymin, xmax, ymax] = bbox;
    return {
      left: `${(xmin / 5.12).toFixed(1)}%`,
      top: `${(ymin / 5.12).toFixed(1)}%`,
      width: `${((xmax - xmin) / 5.12).toFixed(1)}%`,
      height: `${((ymax - ymin) / 5.12).toFixed(1)}%`
    };
  };

  return (
    <div style={{ position: 'relative' }}>
      {/* Dynamic Glowing Spotlight follows Mouse */}
      <div 
        className="interactive-glow" 
        style={{ transform: `translate3d(${glowPos.x}px, ${glowPos.y}px, 0)` }} 
      />

      {/* Viewport Scroll Progress indicator */}
      <div 
        className="scroll-progress-bar" 
        style={{ width: `${scrollPercent}%` }} 
      />

      {/* Background Layer with gradients & bubbles */}
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      {/* Main Glass Dashboard Layout */}
      <div className="app-container">
        {/* Sticky Sidebar Navigation */}
        <aside className="sidebar">
          <div>
            <div className="brand-section">
              <div className="brand-logo">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                </svg>
              </div>
              <span className="brand-name">Intelligent Medical<br/>Image Analysis</span>
            </div>
            
            <nav className="nav-links">
              <div 
                className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
                onClick={() => setActiveTab('dashboard')}
              >
                <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                  <rect width="7" height="9" x="3" y="3" rx="1"/>
                  <rect width="7" height="5" x="14" y="3" rx="1"/>
                  <rect width="7" height="9" x="14" y="12" rx="1"/>
                  <rect width="7" height="5" x="3" y="16" rx="1"/>
                </svg>
                Diagnostics
              </div>
              
              <div 
                className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
                onClick={() => setActiveTab('history')}
              >
                <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                  <path d="M12 8v4l3 3m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/>
                </svg>
                History Archives
              </div>
            </nav>
          </div>

          <div className="sidebar-footer">
            Clinic-ready Diagnostic Suite<br/>
            Zero-Shot SAM 2 & BiomedCLIP.
          </div>
        </aside>

        {/* Scroll Content Area */}
        <main className="main-content">
          <div className="main-content-inner">
            {activeTab === 'dashboard' ? (
              <div ref={dashboardRef}>
                <header className="page-header">
                  <h2 className="page-title">Interactive Analysis Interface</h2>
                  <p className="page-subtitle">Input patient details and upload medical images to route pipeline execution.</p>
                </header>

                  <div className="dashboard-grid">
                    {/* Intake Form Panel */}
                    <div className="glass-panel upload-panel">
                      <form onSubmit={handleAnalyze}>
                        <h3 className="form-title">
                          <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                            <path d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                          </svg>
                          1. Patient Demographics
                        </h3>
                        
                        <div className="form-grid">
                          <div className="form-group">
                            <label htmlFor="pname">Patient Name</label>
                            <input 
                              type="text" 
                              id="pname" 
                              placeholder="John Doe" 
                              value={name} 
                              onChange={(e) => setName(e.target.value)} 
                              required 
                            />
                          </div>
                          
                          <div className="form-group">
                            <label htmlFor="page">Patient Age</label>
                            <input 
                              type="number" 
                              id="page" 
                              placeholder="42" 
                              value={age} 
                              onChange={(e) => setAge(e.target.value)} 
                              required 
                            />
                          </div>
                          
                          <div className="form-group">
                            <label htmlFor="psex">Patient Sex</label>
                            <select 
                              id="psex" 
                              value={sex} 
                              onChange={(e) => setSex(e.target.value)}
                            >
                              <option value="Male">Male</option>
                              <option value="Female">Female</option>
                              <option value="Other">Other</option>
                            </select>
                          </div>
                        </div>

                        <div className="form-grid-2" style={{marginBottom: '30px'}}>
                          <div className="form-group">
                            <label htmlFor="psym">Reported Symptoms</label>
                            <textarea 
                              id="psym" 
                              rows="2" 
                              placeholder="Describe any lesion, pain, discoloration or history details..." 
                              value={symptoms} 
                              onChange={(e) => setSymptoms(e.target.value)} 
                              required
                            ></textarea>
                          </div>

                          <div className="form-group">
                            <label htmlFor="phist">Medical History</label>
                            <textarea 
                              id="phist" 
                              rows="2" 
                              placeholder="No pre-existing history reported" 
                              value={history} 
                              onChange={(e) => setHistory(e.target.value)}
                            ></textarea>
                          </div>
                        </div>

                        <h3 className="form-title">
                          <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                            <path d="M2.25 15.75a6.002 6.002 0 0 0 11.601 2.462m0 0a6.002 6.002 0 0 1-11.601-2.462m11.601 2.462H20.25m0 0a3 3 0 0 0-3-3H9.75M8.25 21h8.25" />
                          </svg>
                          2. Clinical Image File
                        </h3>
                        <div 
                          className={`dropzone ${dragActive ? 'active' : ''}`}
                          onDragEnter={handleDrag}
                          onDragLeave={handleDrag}
                          onDragOver={handleDrag}
                          onDrop={handleDrop}
                          onClick={() => fileInputRef.current.click()}
                        >
                          <input 
                            type="file" 
                            ref={fileInputRef} 
                            style={{display: 'none'}} 
                            accept="image/*"
                            onChange={handleFileChange}
                          />
                          <div className="upload-icon">
                            <svg width="40" height="40" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                              <path d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75A2.25 2.25 0 0 0 16.5 4.5H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Z" />
                            </svg>
                          </div>
                          {selectedFile ? (
                            <div>
                              <p style={{fontWeight: '700', color: 'var(--text-light)'}}>{selectedFile.name}</p>
                              <p style={{fontSize: '12px', color: 'var(--text-muted)'}}>{(selectedFile.size / 1024).toFixed(1)} KB</p>
                            </div>
                          ) : (
                            <div>
                              <p style={{fontWeight: '700'}}>Drag & Drop Image here</p>
                              <p style={{fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px'}}>Supports MRIs, Scans, CTs, & Visible Light photos</p>
                            </div>
                          )}
                        </div>

                        <button 
                          type="submit" 
                          className="btn-primary" 
                          style={{marginTop: '30px'}}
                          disabled={loading}
                        >
                          {loading ? 'Processing neural layers...' : 'Execute Diagnostic Pipeline'}
                        </button>
                      </form>
                    </div>

                    {/* Visualizer Panel */}
                    <div className="glass-panel preview-panel" ref={resultsRef}>
                      <h3 className="form-title" style={{marginBottom: '0'}}>Analysis Visualizer Panel</h3>
                      
                      {currentCase ? (
                        <div style={{display: 'flex', flexDirection: 'column', gap: '24px'}}>
                          <div className="visualizer-container">
                            <img 
                              src={`${UPLOADS_BASE}/${currentCase.image_filename}`} 
                              alt="Scan Preview" 
                              className="visualizer-image"
                            />
                            {/* Phase 2: 3-way overlay rendering */}
                            {imageView === 'mask' && currentCase.mask_filename && (
                              <img
                                src={`${UPLOADS_BASE}/${currentCase.mask_filename}`}
                                alt="SAM 2 Segmentation Mask"
                                className="visualizer-mask"
                              />
                            )}
                            {imageView === 'heatmap' && currentCase.gradcam_filename && (
                              <img
                                src={`${UPLOADS_BASE}/${currentCase.gradcam_filename}`}
                                alt="AI Explainability Heatmap"
                                className="visualizer-mask"
                                style={{ opacity: 0.90 }}
                              />
                            )}
                            {imageView === 'mask' && currentCase.bbox && currentCase.bbox.length === 4 && (
                              <div
                                className="visualizer-box"
                                style={getBBoxStyle(currentCase.bbox)}
                                data-label={`${currentCase.disease} (${(currentCase.confidence * 100).toFixed(0)}%)`}
                              />
                            )}
                          </div>

                          {/* Phase 2: 3-way Image Mode Switcher */}
                          <div className="toggle-container" style={{flexWrap: 'wrap', gap: '8px'}}>
                            <span style={{fontSize: '14px', fontWeight: '600', minWidth: '120px'}}>Image View Mode</span>
                            <div style={{display: 'flex', gap: '6px', flexWrap: 'wrap'}}>
                              {[
                                { key: 'original', label: 'Original' },
                                { key: 'mask',     label: 'SAM Mask' },
                                { key: 'heatmap',  label: '🔥 AI Heatmap' },
                              ].map(opt => (
                                <button
                                  key={opt.key}
                                  onClick={() => setImageView(opt.key)}
                                  style={{
                                    padding: '6px 14px',
                                    borderRadius: '8px',
                                    border: imageView === opt.key ? '2px solid var(--primary)' : '2px solid rgba(255,255,255,0.15)',
                                    background: imageView === opt.key ? 'rgba(14,165,233,0.2)' : 'rgba(255,255,255,0.05)',
                                    color: imageView === opt.key ? 'var(--primary)' : 'var(--text-muted)',
                                    fontWeight: imageView === opt.key ? '700' : '500',
                                    fontSize: '13px',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s'
                                  }}
                                >
                                  {opt.label}
                                </button>
                              ))}
                            </div>
                            {imageView === 'heatmap' && !currentCase.gradcam_filename && (
                              <span style={{fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', width: '100%'}}>
                                ⚠ Heatmap not available for this case (grad-cam library may not be installed).
                              </span>
                            )}
                          </div>

                          {/* Metrics Grid */}
                          <div className="metrics-grid">
                            <div className="metric-card">
                              <span className="metric-label">Modality Type</span>
                              <span className="metric-value">{currentCase.modality} ({currentCase.body_part_scan_type})</span>
                            </div>
                            <div className="metric-card">
                              <span className="metric-label">Pathology Diagnosis</span>
                              <span className="metric-value" style={{color: 'var(--primary)'}}>{currentCase.disease}</span>
                            </div>
                            <div className="metric-card">
                              <span className="metric-label">Classification Confidence</span>
                              <span className="metric-value">{(currentCase.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="metric-card">
                              <span className="metric-label">Calculated Severity</span>
                              <span className={`severity-tag severity-${currentCase.severity.toLowerCase()}-tag`}>
                                {currentCase.severity.toUpperCase()}
                              </span>
                            </div>
                          </div>

                          {/* Recommendation Download Footer */}
                          <div className="toggle-container">
                            <div>
                              <p style={{fontSize: '10.5px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700'}}>Clinical Specialist</p>
                              <p style={{fontSize: '16.5px', fontWeight: '800', color: 'var(--text-light)', marginTop: '2px'}}>{currentCase.specialist}</p>
                            </div>
                            <a 
                              href={`${API_BASE}/reports/${currentCase.id}/pdf`}
                              target="_blank"
                              rel="noreferrer"
                              className="btn-secondary"
                              style={{display: 'inline-flex', alignItems: 'center', gap: '8px', textDecoration: 'none'}}
                            >
                              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                                <path d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                              </svg>
                              Export PDF Report
                            </a>
                          </div>

                          {/* RAG Chat drawer */}
                          <div className="glass-panel rag-drawer" style={{background: 'rgba(255, 255, 255, 0.005)'}}>
                            <h4 style={{fontSize: '14.5px', color: 'var(--secondary)', fontWeight: '700', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: 6}}>
                              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                                <path d="M9.813 15.904 9 21l-.813-5.096A4 4 0 0 0 5.096 12.813L0 12l5.096-.813a4 4 0 0 0 3.091-3.091L9 3l.813 5.096a4 4 0 0 0 3.091 3.091L18 12l-5.096.813a4 4 0 0 0-3.091 3.091Z" />
                              </svg>
                              Guideline Assistant (RAG)
                            </h4>
                            <div className="chat-box">
                              {chatMessages.length === 0 ? (
                                <p style={{color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', margin: 'auto'}}>
                                  Consult clinic vectors about treatment rules.
                                </p>
                              ) : (
                                chatMessages.map((msg, i) => (
                                  <div key={i} className={`chat-msg ${msg.role}`}>
                                    {msg.content.split('\n').map((line, k) => (
                                      <p key={k} style={{marginBottom: line.startsWith('-') ? '4px' : '8px'}}>{line}</p>
                                    ))}
                                  </div>
                                ))
                              )}
                              {chatLoading && <p style={{fontSize: '12.5px', color: 'var(--primary)'}}>Searching vectors...</p>}
                              <div ref={chatEndRef} />
                            </div>
                            <form onSubmit={handleChatSend} className="chat-input-area">
                              <input 
                                type="text" 
                                className="chat-input"
                                placeholder="Query clinical guidelines..."
                                value={chatQuery}
                                onChange={(e) => setChatQuery(e.target.value)}
                              />
                              <button type="submit" className="btn-secondary" style={{padding: '0 18px'}} disabled={chatLoading}>Query</button>
                            </form>
                          </div>

                        </div>
                      ) : (
                        <div style={{textAlign: 'center', color: 'var(--text-muted)', padding: '80px 20px'}}>
                          <div style={{fontSize: '48px', marginBottom: '16px', opacity: '0.4'}}>
                            <svg width="44" height="44" fill="none" stroke="currentColor" strokeWidth="1.2" viewBox="0 0 24 24" style={{margin: '0 auto'}}>
                              <path d="M2.25 15a4.5 4.5 0 0 0 4.5 4.5H18a3.75 3.75 0 0 0 1.332-7.257 3 3 0 0 0-3.758-3.848 5.25 5.25 0 0 0-10.233 2.33A4.502 4.502 0 0 0 2.25 15Z" />
                            </svg>
                          </div>
                          <p style={{fontSize: '15px', fontWeight: '600'}}>Ready for Diagnostic Execution</p>
                          <p style={{fontSize: '12px', marginTop: '6px', maxWidth: '280px', margin: '6px auto 0'}}>Upload an image on the left and enter demographics to run neural inference.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
              /* Historical reports view */
              <div className="reveal-on-scroll">
                <header className="page-header">
                  <h1 className="page-title">Patient Archives & History</h1>
                  <p className="page-subtitle">View, review, and consult RAG guidelines on past analyzed patient records.</p>
                </header>

                <div className="glass-panel" style={{padding: '40px'}}>
                  {/* Phase 2: Search + Filter Bar */}
                  <div style={{display: 'flex', gap: '12px', marginBottom: '24px', flexWrap: 'wrap', alignItems: 'center'}}>
                    <h3 className="form-title" style={{margin: 0, flex: '1 1 200px'}}>Archived Diagnostics Database</h3>
                    <input
                      type="text"
                      placeholder="Search patient name, disease..."
                      value={historySearch}
                      onChange={e => setHistorySearch(e.target.value)}
                      style={{
                        flex: '1 1 200px', padding: '8px 14px', borderRadius: '10px',
                        background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.15)',
                        color: 'var(--text-light)', fontSize: '13px', outline: 'none'
                      }}
                    />
                    <select
                      value={severityFilter}
                      onChange={e => setSeverityFilter(e.target.value)}
                      style={{
                        padding: '8px 14px', borderRadius: '10px',
                        background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.15)',
                        color: 'var(--text-light)', fontSize: '13px', cursor: 'pointer'
                      }}
                    >
                      <option value="all">All Severities</option>
                      <option value="mild">Mild</option>
                      <option value="moderate">Moderate</option>
                      <option value="severe">Severe</option>
                    </select>
                  </div>
                  {(() => {
                    const q = historySearch.toLowerCase();
                    const filtered = reportsHistory.filter(rep => {
                      const matchSearch = !q ||
                        (rep.patient_name || '').toLowerCase().includes(q) ||
                        (rep.disease || '').toLowerCase().includes(q) ||
                        (rep.body_part_scan_type || '').toLowerCase().includes(q);
                      const matchSeverity = severityFilter === 'all' ||
                        (rep.severity || '').toLowerCase() === severityFilter;
                      return matchSearch && matchSeverity;
                    });
                    return filtered.length === 0 ? (
                      <p style={{color: 'var(--text-muted)', padding: '20px 0'}}>No records match your search.</p>
                    ) : (
                    <div className="report-list">
                      {filtered.map((rep) => (
                        <div key={rep.id} className="glass-panel report-item" style={{background: 'rgba(255,255,255,0.01)'}}>
                          <div className="report-meta">
                            <span className="report-title-text">{rep.patient_name} (Age {rep.patient_age}, {rep.patient_sex})</span>
                            <span style={{fontSize: '13px', color: 'var(--primary)', fontWeight: '600'}}>
                              {rep.modality} - {rep.body_part_scan_type} | {rep.disease} ({rep.severity})
                            </span>
                            <span className="report-date">Created: {new Date(rep.created_at).toLocaleString()}</span>
                          </div>
                          
                          <div className="report-actions">
                            <button 
                              className="btn-secondary"
                              onClick={() => selectHistoricCase(rep)}
                            >
                              Restore Case Visuals
                            </button>
                            
                            <a 
                              href={`${API_BASE}/reports/${rep.id}/pdf`}
                              target="_blank"
                              rel="noreferrer"
                              className="btn-secondary"
                              style={{display: 'inline-flex', alignItems: 'center', gap: '6px', textDecoration: 'none'}}
                            >
                              <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                                <path d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                              </svg>
                              Export PDF
                            </a>
                          </div>
                        </div>
                      ))}
                    </div>
                    );
                  })()}
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
