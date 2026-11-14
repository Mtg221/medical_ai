import React, { useEffect, useId, useState } from "react";
import { apiRequest } from "./api.js";

const COPY = {
  en: {
    title: "MedResearch AI Research Platform",
    subtitle: "Bilingual (EN/FR) Medical Extraction, Random-Init AI & Dermatology CV",
    safety: "RESEARCH & INFORMATION EXTRACTION ONLY — NOT FOR AUTONOMOUS CLINICAL DIAGNOSIS OR TREATMENT.",
    skip: "Skip to main content",
    signOut: "Sign out",
    signIn: "Sign in",
    username: "Username",
    password: "Password",
    nav: {
      dashboard: "Overview",
      documents: "Documents",
      ingest: "Ingest",
      timeline: "Timeline",
      search: "Search & Terms",
      benchmarks: "6 Models & Derm CV",
      audit: "Audit Log",
    },
  },
  fr: {
    title: "Plateforme MedResearch IA",
    subtitle: "Extraction médicale bilingue (EN/FR), IA sans pré-entraînement & Dermatologie CV",
    safety: "PLATEFORME DE RECHERCHE UNIQUEMENT — NON DESTINÉE AU DIAGNOSTIC OU AU TRAITEMENT MÉDICAL AUTONOME.",
    skip: "Aller au contenu principal",
    signOut: "Déconnexion",
    signIn: "Connexion",
    username: "Identifiant",
    password: "Mot de passe",
    nav: {
      dashboard: "Vue d'ensemble",
      documents: "Documents",
      ingest: "Ingestion",
      timeline: "Chronologie",
      search: "Recherche & Termes",
      benchmarks: "6 Modèles & Derm CV",
      audit: "Journal d'audit",
    },
  },
};

function ResponsiveTable({ caption, columns, rows }) {
  const capId = useId();
  return (
    <div tabIndex={0} role="region" aria-labelledby={capId} className="table-wrapper">
      <table>
        <caption id={capId}>{caption}</caption>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.key} scope="col">{col.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={row.id || idx}>
              {columns.map((col) => (
                <td key={col.key} data-label={col.label}>
                  {col.render ? col.render(row) : String(row[col.key] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState(() => localStorage.getItem("lang") || "en");
  const [session, setSession] = useState(() => {
    try {
      return JSON.parse(sessionStorage.getItem("session") || "null");
    } catch {
      return null;
    }
  });
  const [view, setView] = useState("dashboard");
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const t = COPY[lang] || COPY.en;
  const token = session?.token || null;
  const user = session?.user || null;
  const can = (perm) => !!user?.permissions?.includes(perm);

  useEffect(() => {
    document.documentElement.lang = lang;
    localStorage.setItem("lang", lang);
  }, [lang]);

  useEffect(() => {
    document.title = `${t.nav[view] || "Workspace"} | ${t.title}`;
  }, [view, t]);

  const logout = () => {
    sessionStorage.removeItem("session");
    setSession(null);
    setView("dashboard");
    setErrorMsg("");
  };

  const callApi = (path, opts = {}) =>
    apiRequest(path, { ...opts, token, onAuthError: logout });

  return (
    <>
      <a href="#main-content" className="skip-link">{t.skip}</a>

      <header className="topbar">
        <div>
          <h1>{t.title}</h1>
          <p className="subtitle">{t.subtitle}</p>
        </div>
        <div className="topbar-controls">
          <label htmlFor="lang-select" className="visually-hidden">Language</label>
          <select id="lang-select" value={lang} onChange={(e) => setLang(e.target.value)}>
            <option value="en">English</option>
            <option value="fr">Français</option>
          </select>
          {user && (
            <>
              <span className="badge">{user.username} ({user.role})</span>
              <button type="button" className="btn btn-secondary" onClick={logout}>
                {t.signOut}
              </button>
            </>
          )}
        </div>
      </header>

      <div className="safety-banner" role="note">{t.safety}</div>

      <div role="status" aria-live="polite" className="visually-hidden">{statusMsg}</div>
      {errorMsg && <div role="alert" className="alert error" style={{ margin: "12px 24px 0" }}>{errorMsg}</div>}

      {!token ? (
        <main id="main-content" tabIndex={-1} style={{ maxWidth: 460, margin: "32px auto", padding: "0 16px" }}>
          <LoginCard
            t={t}
            onSuccess={(next) => {
              sessionStorage.setItem("session", JSON.stringify(next));
              setSession(next);
              setErrorMsg("");
              setStatusMsg("Signed in");
            }}
            onError={setErrorMsg}
          />
        </main>
      ) : (
        <div className="shell">
          <aside className="sidebar">
            <nav aria-label="Primary">
              <ul>
                {Object.entries(t.nav).map(([key, label]) => (
                  <li key={key}>
                    <button
                      type="button"
                      className="nav-btn"
                      aria-current={view === key ? "page" : undefined}
                      onClick={() => {
                        setErrorMsg("");
                        setView(key);
                      }}
                    >
                      {label}
                    </button>
                  </li>
                ))}
              </ul>
            </nav>
          </aside>

          <main id="main-content" tabIndex={-1} className="main-area">
            {view === "dashboard" && <DashboardView callApi={callApi} setError={setErrorMsg} />}
            {view === "documents" && <DocumentsView callApi={callApi} can={can} setError={setErrorMsg} />}
            {view === "ingest" && <IngestView callApi={callApi} can={can} setError={setErrorMsg} />}
            {view === "timeline" && <TimelineView callApi={callApi} can={can} setError={setErrorMsg} />}
            {view === "search" && <SearchView callApi={callApi} can={can} setError={setErrorMsg} />}
            {view === "benchmarks" && <BenchmarksView callApi={callApi} can={can} setError={setErrorMsg} />}
            {view === "audit" && <AuditView callApi={callApi} can={can} setError={setErrorMsg} />}
          </main>
        </div>
      )}
    </>
  );
}

function LoginCard({ t, onSuccess, onError }) {
  const [username, setUsername] = useState("ml_engineer");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    onError("");
    try {
      const data = await apiRequest("/api/v1/auth/token", {
        method: "POST",
        body: { username, password },
      });
      onSuccess({
        token: data.access_token,
        user: {
          user_id: data.user_id,
          username,
          institution_id: data.institution_id,
          role: data.role,
          permissions: data.permissions || [],
        },
      });
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card">
      <h2>{t.signIn}</h2>
      <p className="muted">Enter your institutional credentials (demo accounts seeded via <code>MEDRESEARCH_DEMO_PASSWORD</code>).</p>
      <form onSubmit={submit}>
        <div className="form-field">
          <label htmlFor="login-user">{t.username}</label>
          <input id="login-user" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required />
        </div>
        <div className="form-field">
          <label htmlFor="login-pass">{t.password}</label>
          <input id="login-pass" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
        </div>
        <button type="submit" className="btn" disabled={busy}>
          {busy ? "..." : t.signIn}
        </button>
      </form>
    </section>
  );
}

function DashboardView({ callApi, setError }) {
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);

  useEffect(() => {
    Promise.all([callApi("/api/v1/health"), callApi("/api/v1/models")])
      .then(([h, m]) => {
        setHealth(h);
        setModels(m.models || []);
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <section>
      <h2>Platform Overview</h2>
      {health && (
        <div className="grid-cards">
          <div className="stat-card">
            <div className="stat-label">Execution Mode</div>
            <div className="stat-value">{health.hardware?.execution_mode || "CPU-ONLY"}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Pretrained Weights</div>
            <div className="stat-value">{health.pretrained_weights_allowed ? "Allowed" : "Strictly Disabled"}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Registered Models</div>
            <div className="stat-value">{models.length}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Database Backend</div>
            <div className="stat-value">{health.database_backend}</div>
          </div>
        </div>
      )}

      {models.length > 0 && (
        <div className="card">
          <h3>Registered Random-Init Models</h3>
          <ResponsiveTable
            caption="Registered primary research models (zero pretrained weights)"
            columns={[
              { key: "model_id", label: "Model ID" },
              { key: "architecture_name", label: "Architecture" },
              { key: "parameter_count", label: "Params" },
              { key: "pretrained_weights_used", label: "Pretrained", render: (r) => (r.pretrained_weights_used ? "Yes" : "No") },
            ]}
            rows={models}
          />
        </div>
      )}
    </section>
  );
}

function DocumentsView({ callApi, can, setError }) {
  const [docs, setDocs] = useState([]);
  const [selected, setSelected] = useState(null);

  const loadDocs = () => {
    callApi("/api/v1/documents")
      .then((d) => setDocs(d.documents || []))
      .catch((e) => setError(e.message));
  };

  useEffect(() => {
    loadDocs();
  }, []);

  const openDoc = async (docId) => {
    try {
      const detail = await callApi(`/api/v1/documents/${docId}`);
      setSelected(detail);
    } catch (e) {
      setError(e.message);
    }
  };

  const reviewEntity = async (entityId, action) => {
    try {
      await callApi(`/api/v1/entities/${entityId}/review`, {
        method: "POST",
        body: { action },
      });
      if (selected?.document?.document_id) {
        await openDoc(selected.document.document_id);
      }
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section>
      <h2>Medical Documents & Human Review</h2>
      <div className="card">
        <ResponsiveTable
          caption="Ingested medical documents"
          columns={[
            { key: "document_id", label: "Document ID" },
            { key: "language", label: "Lang" },
            { key: "document_type", label: "Type" },
            { key: "specialty", label: "Specialty" },
            {
              key: "actions",
              label: "Inspect",
              render: (r) => (
                <button type="button" className="btn btn-secondary" onClick={() => openDoc(r.document_id)}>
                  Open
                </button>
              ),
            },
          ]}
          rows={docs}
        />
      </div>

      {selected && (
        <div className="card">
          <h3>Extracted Entities — {selected.document?.document_id}</h3>
          <ResponsiveTable
            caption="Extracted entities with provenance and human validation"
            columns={[
              { key: "source_text", label: "Text" },
              { key: "entity_type", label: "Type" },
              { key: "assertion_status", label: "Assertion" },
              { key: "temporal_status", label: "Temporal" },
              { key: "terminology_concept_id", label: "Concept" },
              { key: "confidence", label: "Conf" },
              { key: "review_status", label: "Status" },
              {
                key: "review",
                label: "Review",
                render: (e) =>
                  can("annotation:review") ? (
                    <div className="btn-row">
                      <button type="button" className="btn btn-secondary" onClick={() => reviewEntity(e.entity_id, "APPROVE")}>Approve</button>
                      <button type="button" className="btn btn-secondary" onClick={() => reviewEntity(e.entity_id, "REJECT")}>Reject</button>
                    </div>
                  ) : (
                    <span className="muted">Read-only</span>
                  ),
              },
            ]}
            rows={selected.entities || []}
          />
        </div>
      )}
    </section>
  );
}

function IngestView({ callApi, can, setError }) {
  const [subjectId, setSubjectId] = useState("subj_react_demo");
  const [filename, setFilename] = useState("clinical_note.txt");
  const [text, setText] = useState(
    "Patient presents with chest pain and fever. Prescribed metformin 500 mg twice daily orally. Pulmonary embolism ruled out."
  );
  const [fileBase64, setFileBase64] = useState("");
  const [isImage, setIsImage] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [docDetail, setDocDetail] = useState(null);
  const [visionOut, setVisionOut] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) {
      setFileBase64("");
      setIsImage(false);
      setPreviewUrl("");
      return;
    }
    setFilename(file.name);
    const imgFlag = file.type.startsWith("image/") || /\.(png|jpe?g|tiff?|webp)$/i.test(file.name);
    setIsImage(imgFlag);
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const commaIdx = dataUrl.indexOf(",");
      setFileBase64(commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : "");
      setPreviewUrl(imgFlag ? dataUrl : "");
    };
    reader.readAsDataURL(file);
  };

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    setDocDetail(null);
    setVisionOut(null);
    try {
      const payload = fileBase64
        ? { subject_id: subjectId, filename, content_base64: fileBase64 }
        : { subject_id: subjectId, filename, text };
      const res = await callApi("/api/v1/documents/ingest", {
        method: "POST",
        body: payload,
      });
      setResult(res);

      const detailPromise = callApi(`/api/v1/documents/${encodeURIComponent(res.document_id)}`);
      const visionPromise = (fileBase64 && isImage)
        ? callApi("/api/v1/vision/analyze", { method: "POST", body: { content_base64: fileBase64 } })
        : Promise.resolve(null);

      const [detail, vision] = await Promise.all([detailPromise, visionPromise]);
      setDocDetail(detail);
      setVisionOut(vision);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section>
      <h2>Ingest Medical Document or Image</h2>
      <div className="card">
        <form onSubmit={submit}>
          <div className="form-field">
            <label htmlFor="ingest-subj">Subject ID</label>
            <input id="ingest-subj" value={subjectId} onChange={(e) => setSubjectId(e.target.value)} required />
          </div>
          <div className="form-field">
            <label htmlFor="ingest-upload">Upload Picture or Document (PNG, JPG, WEBP, TIFF, PDF, DOCX, TXT)</label>
            <input
              id="ingest-upload"
              type="file"
              accept=".png,.jpg,.jpeg,.tif,.tiff,.webp,.pdf,.docx,.txt,image/*"
              onChange={handleFileChange}
            />
          </div>
          {previewUrl && (
            <div style={{ marginBottom: 12 }}>
              <img
                src={previewUrl}
                alt={`Preview of ${filename}`}
                style={{ maxWidth: 260, maxHeight: 200, borderRadius: 8, border: "1px solid var(--border)" }}
              />
            </div>
          )}
          <div className="form-field">
            <label htmlFor="ingest-file">Filename</label>
            <input id="ingest-file" value={filename} onChange={(e) => setFilename(e.target.value)} required />
          </div>
          {!fileBase64 && (
            <div className="form-field">
              <label htmlFor="ingest-text">Or Paste Clinical Text (EN or FR)</label>
              <textarea id="ingest-text" rows={5} value={text} onChange={(e) => setText(e.target.value)} required />
            </div>
          )}
          <div className="btn-row">
            <button type="submit" className="btn" disabled={!can("medical_data:write") || busy}>
              {busy ? "Uploading..." : "Ingest & Extract"}
            </button>
            {fileBase64 && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  setFileBase64("");
                  setIsImage(false);
                  setPreviewUrl("");
                  setFilename("clinical_note.txt");
                }}
              >
                Clear File
              </button>
            )}
          </div>
        </form>
      </div>
      {result && (
        <div className="alert ok">
          Ingested <strong>{result.document_id}</strong> ({result.file_format || result.detected_language}) — extracted {result.entities_extracted ?? result.entity_count ?? 0} entities.
        </div>
      )}

      {visionOut && (
        <div className="card">
          <h3>Vision &amp; Dermatology Model Output</h3>
          <div className="grid-cards">
            <div className="stat-card">
              <div className="stat-label">Predicted Modality</div>
              <div className="stat-value" style={{ fontSize: "1.05rem" }}>{visionOut.predicted_modality}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Quality Score</div>
              <div className="stat-value">{visionOut.quality_score}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Derm Group (Random-Init)</div>
              <div className="stat-value">{visionOut.derm_predicted_group}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Probabilities</div>
              <div className="stat-value" style={{ fontSize: "0.95rem" }}>
                benign: {visionOut.derm_probabilities?.benign} · malignant: {visionOut.derm_probabilities?.malignant}
              </div>
            </div>
          </div>
          <p className="muted">
            Resolution: {visionOut.width}×{visionOut.height} | ROI bbox: <code>{JSON.stringify(visionOut.roi_bbox)}</code> | Pretrained: {String(visionOut.pretrained_weights_used)} — {visionOut.disclaimer}
          </p>
        </div>
      )}

      {docDetail && (
        <div className="card">
          <h3>Model Extraction Output ({docDetail.document?.document_id})</h3>
          {docDetail.pages?.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <strong>Parsed / OCR Text:</strong>
              <pre style={{ background: "var(--surface-alt)", padding: 10, borderRadius: 6, whiteSpace: "pre-wrap", fontSize: "0.86rem" }}>
                {docDetail.pages.map((p) => p.raw_text).join("\n\n")}
              </pre>
            </div>
          )}
          {docDetail.entities?.length > 0 ? (
            <ResponsiveTable
              caption="Extracted medical entities, assertions, temporality, and terminology mapping"
              columns={[
                { key: "source_text", label: "Extracted Text" },
                { key: "entity_type", label: "Entity Type" },
                { key: "assertion_status", label: "Assertion" },
                { key: "temporal_status", label: "Temporality" },
                { key: "terminology_concept_id", label: "Concept ID" },
                { key: "confidence", label: "Confidence" },
              ]}
              rows={docDetail.entities}
            />
          ) : (
            <p className="muted">No clinical text entities detected in this upload.</p>
          )}
          {docDetail.relations?.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <ResponsiveTable
                caption="Extracted medical relationships"
                columns={[
                  { key: "relation_type", label: "Relation" },
                  { key: "evidence_text", label: "Evidence" },
                  { key: "confidence", label: "Confidence" },
                ]}
                rows={docDetail.relations}
              />
            </div>
          )}
          <p className="muted" style={{ marginTop: 10 }}>{docDetail.clinical_disclaimer}</p>
        </div>
      )}
    </section>
  );
}

function TimelineView({ callApi, can, setError }) {
  const [subjectId, setSubjectId] = useState("subj_synth_001");
  const [data, setData] = useState(null);

  const loadTimeline = async (e) => {
    if (e) e.preventDefault();
    setError("");
    try {
      const res = await callApi(`/api/v1/subjects/${encodeURIComponent(subjectId)}/timeline`);
      setData(res);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <section>
      <h2>Longitudinal Subject Timeline</h2>
      <div className="card">
        <form onSubmit={loadTimeline} className="btn-row">
          <label htmlFor="timeline-subj" className="visually-hidden">Subject ID</label>
          <input id="timeline-subj" value={subjectId} onChange={(e) => setSubjectId(e.target.value)} required />
          <button type="submit" className="btn" disabled={!can("medical_data:read")}>
            Load Timeline
          </button>
        </form>
      </div>
      {data && (
        <div className="card">
          <h3>Events ({(data.timeline_events || []).length})</h3>
          <ResponsiveTable
            caption={`Longitudinal timeline for ${subjectId}`}
            columns={[
              { key: "encounter_date_iso", label: "Date" },
              { key: "event_type", label: "Event" },
              { key: "summary_text", label: "Summary" },
              { key: "document_id", label: "Document" },
            ]}
            rows={data.timeline_events || []}
          />
          {(data.contradictions || []).length > 0 && (
            <div className="alert warn" style={{ marginTop: 12 }}>
              Detected {data.contradictions.length} cross-document contradiction(s).
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function SearchView({ callApi, can, setError }) {
  const [query, setQuery] = useState("metformin");
  const [hits, setHits] = useState([]);
  const [term, setTerm] = useState("hypertension artérielle");
  const [termRes, setTermRes] = useState(null);

  const runSearch = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const res = await callApi(`/api/v1/search?q=${encodeURIComponent(query)}`);
      setHits(res.results || []);
    } catch (err) {
      setError(err.message);
    }
  };

  const runMap = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const res = await callApi(`/api/v1/terminology/map?term=${encodeURIComponent(term)}`);
      setTermRes(res);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <section>
      <h2>Hybrid Search & Bilingual Terminology</h2>
      <div className="card">
        <h3>Hybrid Document Search</h3>
        <form onSubmit={runSearch} className="btn-row">
          <label htmlFor="search-q" className="visually-hidden">Search query</label>
          <input id="search-q" value={query} onChange={(e) => setQuery(e.target.value)} required />
          <button type="submit" className="btn" disabled={!can("search:query")}>Search</button>
        </form>
        {hits.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <ResponsiveTable
              caption="Search results"
              columns={[
                { key: "document_id", label: "Document" },
                { key: "score", label: "Score" },
                { key: "snippet", label: "Snippet" },
              ]}
              rows={hits}
            />
          </div>
        )}
      </div>

      <div className="card">
        <h3>Terminology Normalization (ICD-10 / SNOMED-CT / RxNorm / LOINC)</h3>
        <form onSubmit={runMap} className="btn-row">
          <label htmlFor="term-q" className="visually-hidden">Medical term</label>
          <input id="term-q" value={term} onChange={(e) => setTerm(e.target.value)} required />
          <button type="submit" className="btn btn-accent">Map Term</button>
        </form>
        {termRes && (
          <div className="alert ok" style={{ marginTop: 12 }}>
            <strong>{termRes.concept_id || "UNMAPPED"}</strong> ({termRes.vocabulary} {termRes.code}) — {termRes.match_type} (conf {termRes.confidence})
          </div>
        )}
      </div>
    </section>
  );
}

function BenchmarksView({ callApi, can, setError }) {
  const [sixData, setSixData] = useState(null);
  const [dermData, setDermData] = useState(null);
  const [expList, setExpList] = useState({ experiments: [], federated_rounds: [] });
  const [busy, setBusy] = useState("");

  useEffect(() => {
    callApi("/api/v1/experiments")
      .then((res) => setExpList(res))
      .catch(() => {});
  }, []);

  const runPhase1 = async () => {
    setBusy("phase1");
    setError("");
    try {
      await callApi("/api/v1/experiments/run", { method: "POST" });
      const res = await callApi("/api/v1/experiments");
      setExpList(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy("");
    }
  };

  const runSix = async () => {
    setBusy("six");
    setError("");
    try {
      const res = await callApi("/api/v1/extraction/benchmark", {
        method: "POST",
        body: { epochs: 15, seed: 42 },
      });
      setSixData(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy("");
    }
  };

  const runDerm = async () => {
    setBusy("derm");
    setError("");
    try {
      const res = await callApi("/api/v1/dermatology/benchmark", {
        method: "POST",
        body: { epochs: 15, per_class_limit: 150, seed: 42 },
      });
      setDermData(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy("");
    }
  };

  return (
    <section>
      <h2>Six Extraction Approaches & Dermatology CV</h2>
      <div className="card">
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" disabled={!can("training:execute") || busy !== ""} onClick={runPhase1}>
            {busy === "phase1" ? "Running Phase 1..." : "Run Phase 1 Baseline"}
          </button>
          <button type="button" className="btn" disabled={!can("training:execute") || busy !== ""} onClick={runSix}>
            {busy === "six" ? "Running 6 Approaches..." : "Run 6-Approach Extraction Benchmark"}
          </button>
          <button type="button" className="btn btn-accent" disabled={!can("training:execute") || busy !== ""} onClick={runDerm}>
            {busy === "derm" ? "Training Derm ConvNet..." : "Run Dermatology CV Benchmark"}
          </button>
        </div>
        {!can("training:execute") && <p className="muted">Requires training:execute permission (ML_ENGINEER or INSTITUTION_ADMIN).</p>}
      </div>

      {expList.experiments?.length > 0 && (
        <div className="card">
          <h3>Stored Experiments ({expList.experiments.length})</h3>
          <ResponsiveTable
            caption="Stored reproducible experiment runs"
            columns={[
              { key: "experiment_id", label: "Experiment ID" },
              { key: "status", label: "Status" },
              { key: "runtime_seconds", label: "Runtime (s)" },
              { key: "peak_rss_mb", label: "Peak RSS (MB)" },
            ]}
            rows={expList.experiments}
          />
        </div>
      )}

      {sixData && (
        <div className="card">
          <h3>Six-Approach Extraction Comparison ({sixData.corpus})</h3>
          <p className="muted">
            Train docs: {sixData.train_documents} | Test docs: {sixData.test_documents} | Leakage verified: {String(sixData.leakage_verified)} | Pretrained: {String(sixData.pretrained_weights_used)}
          </p>
          <ResponsiveTable
            caption="Comparative metrics across all six independent medical extraction approaches"
            columns={[
              { key: "model_name", label: "Model" },
              { key: "precision", label: "Precision" },
              { key: "recall", label: "Recall" },
              { key: "f1", label: "F1" },
              { key: "en_f1", label: "EN F1" },
              { key: "fr_f1", label: "FR F1" },
              { key: "parameter_count", label: "Params" },
              { key: "train_seconds", label: "Train (s)" },
            ]}
            rows={sixData.approaches || []}
          />
        </div>
      )}

      {dermData && (
        <div className="card">
          <h3>Dermatology CV — RandomInitDermConvNet ({dermData.data_source})</h3>
          <div className="alert ok">
            Accuracy: <strong>{dermData.evaluation?.accuracy}</strong> | Brier: <strong>{dermData.evaluation?.brier_score}</strong> | Loss: {dermData.training?.initial_loss} → {dermData.training?.final_loss} | Params: {dermData.training?.parameter_count} | Pretrained: {String(dermData.evaluation?.pretrained_weights_used)}
          </div>
          <ResponsiveTable
            caption="Subgroup fairness accuracy by Fitzpatrick skin tone"
            columns={[
              { key: "tone", label: "Skin Tone" },
              { key: "count", label: "Samples" },
              { key: "accuracy", label: "Accuracy" },
            ]}
            rows={Object.entries(dermData.evaluation?.by_skin_tone || {}).map(([tone, v]) => ({
              tone,
              count: v.count,
              accuracy: v.accuracy,
            }))}
          />
          <p className="muted" style={{ marginTop: 8 }}>{dermData.evaluation?.disclaimer}</p>
        </div>
      )}
    </section>
  );
}

function AuditView({ callApi, can, setError }) {
  const [audit, setAudit] = useState(null);

  const verify = () => {
    setError("");
    callApi("/api/v1/audit/verify")
      .then(setAudit)
      .catch((e) => setError(e.message));
  };

  return (
    <section>
      <h2>Tamper-Evident Audit Chain</h2>
      <div className="card">
        <button type="button" className="btn" disabled={!can("audit:read")} onClick={verify}>
          Verify SHA-256 Audit Chain
        </button>
        {audit && (
          <div className={`alert ${audit.valid ? "ok" : "error"}`} style={{ marginTop: 12 }}>
            {audit.valid
              ? `Valid hash chain — ${audit.checked_entries} entries verified. Head: ${audit.head_hash_sha256}`
              : `Invalid: ${audit.error}`}
          </div>
        )}
      </div>
    </section>
  );
}
