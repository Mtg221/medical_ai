"use strict";

/* Medical documents and model output are untrusted: every dynamic value is rendered with
   textContent via h(); HTML string injection is never used. */

const I18N = {
  en: {
    skip: "Skip to main content", title: "MedResearch Intelligence Platform",
    subtitle: "Bilingual (EN/FR) medical information extraction & research workspace",
    language: "Language", logout: "Sign out", signIn: "Sign in", username: "Username", password: "Password",
    loginTitle: "Sign in to the research workspace", loginFailed: "Sign-in failed",
    dashboard: "Dashboard", documents: "Documents", ingest: "Ingest", timeline: "Timeline", search: "Search",
    terminology: "Terminology", models: "Models", experiments: "Experiments", audit: "Audit",
    loading: "Loading…", noData: "No data", refresh: "Refresh", open: "Open", back: "Back to documents",
    docs: "Documents", entities: "Entities", relations: "Relations", ocr: "OCR blocks", registeredModels: "Registered models",
    auditValid: "Audit chain", valid: "valid", invalid: "INVALID", entries: "entries",
    evidence: "Source text with evidence highlights", legend: "Provenance: AI_PREDICTED = model suggestion, not ground truth. APPROVED_GROUND_TRUTH = human reviewed.",
    review: "Review", decision: "Decision", rationale: "Rationale", submit: "Submit review", cancel: "Cancel",
    correctedAssertion: "Corrected assertion (optional)", correctedTemporal: "Corrected temporal status (optional)",
    keep: "— keep —", reviewSaved: "Review saved", noPermission: "Your role lacks permission for this workspace.",
    filename: "Filename", text: "Document text", orFile: "or upload a file (TXT, PDF, DOCX, PNG, JPEG)",
    subjectId: "Subject ID", familyId: "Document family ID", datasetId: "Dataset ID", encounterDate: "Encounter date",
    split: "Split", specialty: "Specialty", docType: "Document type", ingestBtn: "Ingest & extract", ingested: "Document ingested",
    subject: "Subject", contradictions: "Contradictions", date: "Date", category: "Category", summary: "Summary",
    assertion: "Assertion", temporal: "Temporal", conf: "Conf.", unc: "Unc.", flag: "Flag",
    query: "Query", run: "Run", results: "Results", score: "Score", explanation: "Ranking explanation",
    term: "Term (EN or FR)", mapTerm: "Map term", anyCategory: "Any category",
    runExperiment: "Run Phase 1 benchmark", runningExp: "Running benchmark (CPU only)…", metrics: "Metrics",
    verifyAudit: "Verify audit chain", health: "System", mode: "Execution mode", cpus: "CPUs", ram: "RAM (GB)", version: "Version",
    type: "Type", normalized: "Normalized", med: "Medication", review_status: "Review", id: "ID",
    disclaimer: "Mapped codes are normalized research concepts, not approved diagnoses.",
    confirmRun: "Run the full benchmark now? It retrains the models on CPU.",
  },
  fr: {
    skip: "Aller au contenu principal", title: "Plateforme d'intelligence MedResearch",
    subtitle: "Extraction d'informations médicales bilingue (EN/FR) et espace de recherche",
    language: "Langue", logout: "Se déconnecter", signIn: "Se connecter", username: "Identifiant", password: "Mot de passe",
    loginTitle: "Connexion à l'espace de recherche", loginFailed: "Échec de la connexion",
    dashboard: "Tableau de bord", documents: "Documents", ingest: "Ingestion", timeline: "Chronologie", search: "Recherche",
    terminology: "Terminologie", models: "Modèles", experiments: "Expériences", audit: "Audit",
    loading: "Chargement…", noData: "Aucune donnée", refresh: "Actualiser", open: "Ouvrir", back: "Retour aux documents",
    docs: "Documents", entities: "Entités", relations: "Relations", ocr: "Blocs OCR", registeredModels: "Modèles enregistrés",
    auditValid: "Chaîne d'audit", valid: "valide", invalid: "INVALIDE", entries: "entrées",
    evidence: "Texte source avec preuves surlignées", legend: "Provenance : AI_PREDICTED = suggestion du modèle, pas une vérité terrain. APPROVED_GROUND_TRUTH = validé par un humain.",
    review: "Réviser", decision: "Décision", rationale: "Justification", submit: "Enregistrer la révision", cancel: "Annuler",
    correctedAssertion: "Assertion corrigée (facultatif)", correctedTemporal: "Statut temporel corrigé (facultatif)",
    keep: "— conserver —", reviewSaved: "Révision enregistrée", noPermission: "Votre rôle n'a pas accès à cet espace.",
    filename: "Nom du fichier", text: "Texte du document", orFile: "ou téléverser un fichier (TXT, PDF, DOCX, PNG, JPEG)",
    subjectId: "ID du sujet", familyId: "ID de la famille de documents", datasetId: "ID du jeu de données", encounterDate: "Date de la consultation",
    split: "Partition", specialty: "Spécialité", docType: "Type de document", ingestBtn: "Ingérer et extraire", ingested: "Document ingéré",
    subject: "Sujet", contradictions: "Contradictions", date: "Date", category: "Catégorie", summary: "Résumé",
    assertion: "Assertion", temporal: "Temporalité", conf: "Conf.", unc: "Inc.", flag: "Alerte",
    query: "Requête", run: "Lancer", results: "Résultats", score: "Score", explanation: "Explication du classement",
    term: "Terme (EN ou FR)", mapTerm: "Normaliser", anyCategory: "Toute catégorie",
    runExperiment: "Lancer le benchmark Phase 1", runningExp: "Benchmark en cours (CPU)…", metrics: "Métriques",
    verifyAudit: "Vérifier la chaîne d'audit", health: "Système", mode: "Mode d'exécution", cpus: "CPU", ram: "RAM (Go)", version: "Version",
    type: "Type", normalized: "Normalisé", med: "Médicament", review_status: "Révision", id: "ID",
    disclaimer: "Les codes proposés sont des concepts de recherche normalisés, pas des diagnostics approuvés.",
    confirmRun: "Lancer le benchmark complet ? Les modèles sont ré-entraînés sur CPU.",
  },
};

const state = { token: null, user: null, lang: localStorage.getItem("lang") || "en", view: "dashboard" };

const t = (key) => (I18N[state.lang] && I18N[state.lang][key]) || I18N.en[key] || key;

/* ---------- DOM helpers ---------- */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const view = () => document.getElementById("view");
function mount(...nodes) { const v = view(); v.replaceChildren(...nodes); }
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg; el.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => { el.hidden = true; }, 3500);
}
const alertBox = (kind, msg) => h("div", { class: `alert ${kind}`, role: kind === "error" ? "alert" : "status" }, msg);
const fmt = (n, d = 3) => (typeof n === "number" ? n.toFixed(d) : n ?? "");

function badge(value, kind) {
  const map = { AI_PREDICTED: "ai", APPROVED_GROUND_TRUTH: "gt", HUMAN_CORRECTED: "gt", REJECTED: "bad",
    UNRESOLVED_DISAGREEMENT: "warn", NEGATED: "bad", UNCERTAIN: "warn", HYPOTHETICAL: "warn", FAMILY_HISTORY: "warn" };
  return h("span", { class: `badge ${kind || map[value] || ""}` }, value);
}

/* ---------- API ---------- */
async function api(path, { method = "GET", body } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  let res;
  try {
    res = await fetch(path, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined });
  } catch (e) {
    throw new Error("Network error: " + e.message);
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) {
    const msg = (data && data.error) || `HTTP ${res.status}`;
    if (state.token && (/Invalid access token signature/i.test(msg) || /Token expired/i.test(msg))) {
      logout();
    }
    throw new Error(msg);
  }
  return data;
}
const can = (perm) => !!state.user && state.user.permissions.includes(perm);

/* ---------- Session ---------- */
function saveSession() { sessionStorage.setItem("session", JSON.stringify({ token: state.token, user: state.user })); }
function loadSession() {
  try {
    const s = JSON.parse(sessionStorage.getItem("session") || "null");
    if (s && s.token) { state.token = s.token; state.user = s.user; }
  } catch { sessionStorage.removeItem("session"); }
}
function logout() { state.token = null; state.user = null; sessionStorage.removeItem("session"); boot(); }

/* ---------- Navigation ---------- */
const NAV = [
  { id: "dashboard", perm: null, render: renderDashboard },
  { id: "documents", perm: "medical_data:read", render: renderDocuments },
  { id: "ingest", perm: "medical_data:write", render: renderIngest },
  { id: "timeline", perm: "medical_data:read", render: renderTimeline },
  { id: "search", perm: "search:query", render: renderSearch },
  { id: "terminology", perm: null, render: renderTerminology },
  { id: "models", perm: null, render: renderModels },
  { id: "experiments", perm: null, render: renderExperiments },
  { id: "audit", perm: "audit:read", render: renderAudit },
];

function renderNav() {
  const nav = document.getElementById("nav");
  nav.hidden = !state.token;
  nav.replaceChildren(...NAV.filter((n) => !n.hidden).map((n) =>
    h("button", { type: "button", "aria-current": state.view === n.id ? "page" : null, onclick: () => go(n.id) }, t(n.id))));
}
function go(id, arg) {
  state.view = id; renderNav();
  const entry = NAV.find((n) => n.id === id);
  if (entry.perm && !can(entry.perm)) { mount(h("h2", {}, t(id)), alertBox("info", t("noPermission"))); return; }
  mount(h("p", { class: "muted" }, t("loading")));
  Promise.resolve(entry.render(arg)).catch((e) => mount(h("h2", {}, t(id)), alertBox("error", e.message)));
  view().focus();
}

/* ---------- Views ---------- */
async function renderDashboard() {
  const [health, docs, models] = await Promise.all([
    api("/api/v1/health"),
    can("medical_data:read") ? api("/api/v1/documents") : Promise.resolve(null),
    api("/api/v1/models"),
  ]);
  const hw = health.readiness.hardware;
  let audit = null;
  if (can("audit:read")) audit = await api("/api/v1/audit/verify");
  const stat = (label, value) => h("div", { class: "stat" }, h("div", { class: "num" }, value), h("div", { class: "muted" }, label));
  mount(
    h("h2", {}, t("dashboard")),
    h("div", { class: "grid" },
      stat(t("docs"), docs ? docs.count : "—"),
      stat(t("registeredModels"), models.count),
      stat(t("auditValid"), audit ? `${audit.valid ? t("valid") : t("invalid")} (${audit.checked_entries} ${t("entries")})` : "—"),
      stat(t("mode"), hw.execution_mode)),
    h("section", { class: "card" },
      h("h3", {}, t("health")),
      h("p", {}, `${t("version")}: ${health.version} · ${t("cpus")}: ${hw.physical_cores}/${hw.logical_cpus} · ${t("ram")}: ${hw.total_ram_gb} · ${hw.os_name} ${hw.machine_arch}`),
      health.readiness.warnings.map((w) => h("p", { class: "muted" }, w))),
    h("p", { class: "muted" }, t("legend")));
}

async function renderDocuments() {
  const data = await api("/api/v1/documents");
  const rows = data.documents.map((d) => h("tr", { class: "clickable", tabindex: "0",
    onclick: () => go("document", d.document_id),
    onkeydown: (e) => { if (e.key === "Enter") go("document", d.document_id); } },
    h("td", {}, d.document_id), h("td", {}, d.original_filename), h("td", {}, d.file_format),
    h("td", {}, d.detected_language), h("td", {}, d.specialty), h("td", {}, d.subject_id),
    h("td", {}, JSON.parse(d.untrusted_flags_json || "[]").length ? badge("FLAGGED", "warn") : "")));
  mount(h("h2", {}, t("documents")),
    h("div", { class: "card table-wrap" }, rows.length
      ? h("table", {}, h("thead", {}, h("tr", {}, ["ID", t("filename"), "Format", "Lang", t("specialty"), t("subject"), ""].map((c) => h("th", { scope: "col" }, c)))), h("tbody", {}, rows))
      : h("p", {}, t("noData"))));
}
NAV.push({ id: "document", perm: "medical_data:read", render: renderDocumentDetail, hidden: true });

function entityClass(e) {
  if (e.assertion_status === "NEGATED") return "negated";
  if (e.assertion_status === "UNCERTAIN" || e.assertion_status === "HYPOTHETICAL") return "uncertain";
  if (e.assertion_status === "FAMILY_HISTORY") return "family";
  return "";
}
function highlightedText(text, entities) {
  const sorted = [...entities].sort((a, b) => a.char_start - b.char_start);
  const out = []; let cursor = 0;
  for (const e of sorted) {
    if (e.char_start < cursor || e.char_end > text.length) continue;
    out.push(text.slice(cursor, e.char_start));
    out.push(h("mark", { class: `ent ${entityClass(e)}`, tabindex: "0",
      title: `${e.entity_type} · ${e.assertion_status} · ${e.temporal_status} · conf ${fmt(e.confidence)}` },
      text.slice(e.char_start, e.char_end)));
    cursor = e.char_end;
  }
  out.push(text.slice(cursor));
  return out;
}

async function renderDocumentDetail(docId) {
  const d = await api(`/api/v1/documents/${encodeURIComponent(docId)}`);
  const reviewHost = h("div", {});
  const entRows = d.entities.map((e) => h("tr", {},
    h("td", {}, e.entity_type), h("td", {}, e.source_text), h("td", {}, e.normalized_text + (e.terminology_concept_id ? ` (${e.mapping_match_type})` : "")),
    h("td", {}, badge(e.assertion_status)), h("td", {}, e.temporal_status),
    h("td", {}, e.medication_status === "NOT_APPLICABLE" ? "" : e.medication_status),
    h("td", {}, fmt(e.confidence)), h("td", {}, fmt(e.uncertainty) + (e.abstained ? " ⚠" : "")),
    h("td", {}, badge(e.review_status)),
    h("td", {}, can("annotation:review") ? h("button", { type: "button", class: "btn btn-small", onclick: () => openReview(e, reviewHost, docId) }, t("review")) : "")));
  const entById = Object.fromEntries(d.entities.map((e) => [e.entity_id, e]));
  const relRows = d.relations.map((r) => h("tr", {},
    h("td", {}, entById[r.source_entity_id]?.source_text ?? r.source_entity_id), h("td", {}, r.relation_type),
    h("td", {}, entById[r.target_entity_id]?.source_text ?? r.target_entity_id), h("td", {}, badge(r.derivation_type, "")), h("td", {}, fmt(r.confidence))));
  const ocrRows = d.ocr_blocks.map((b) => h("tr", {},
    h("td", {}, b.block_index), h("td", {}, b.recognized_text), h("td", {}, fmt(b.confidence)),
    h("td", {}, fmt(b.aleatoric_uncertainty)), h("td", {}, fmt(b.epistemic_uncertainty)), h("td", {}, badge(b.review_status))));
  const table = (heads, rows) => rows.length
    ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, heads.map((c) => h("th", { scope: "col" }, c)))), h("tbody", {}, rows)))
    : h("p", { class: "muted" }, t("noData"));
  const flags = JSON.parse(d.document.untrusted_flags_json || "[]");
  mount(
    h("button", { type: "button", class: "btn btn-secondary", onclick: () => go("documents") }, "← " + t("back")),
    h("h2", {}, d.document.original_filename),
    h("p", { class: "muted" }, `${d.document.document_id} · ${d.document.file_format} · ${d.document.detected_language} · SHA-256 ${d.document.content_sha256.slice(0, 16)}…`),
    flags.length ? alertBox("error", "Untrusted content flags: " + flags.join(", ")) : null,
    h("p", { class: "muted" }, t("legend")),
    h("section", { class: "card" }, h("h3", {}, t("evidence")),
      d.pages.map((p) => h("div", {}, h("div", { class: "muted" }, `Page ${p.page_number} · ${p.extraction_source}`),
        h("div", { class: "doc-text" }, highlightedText(p.raw_text, d.entities.filter((e) => e.page_number === p.page_number)))))),
    reviewHost,
    h("section", { class: "card" }, h("h3", {}, `${t("entities")} (${d.entities.length})`),
      table([t("type"), "Text", t("normalized"), t("assertion"), t("temporal"), t("med"), t("conf"), t("unc"), t("review_status"), ""], entRows)),
    h("section", { class: "card" }, h("h3", {}, `${t("relations")} (${d.relations.length})`),
      table(["Source", "Relation", "Target", "Derivation", t("conf")], relRows)),
    h("section", { class: "card" }, h("h3", {}, `${t("ocr")} (${d.ocr_blocks.length})`),
      table(["#", "Text", t("conf"), "Aleatoric", "Epistemic", t("review_status")], ocrRows)),
    h("p", { class: "muted" }, d.clinical_disclaimer));
}

function openReview(entity, host, docId) {
  const decision = h("select", { id: "rv-decision" },
    ["APPROVE_AS_GROUND_TRUTH", "CORRECT_AND_APPROVE", "FLAG_AMBIGUOUS", "REJECT"].map((v) => h("option", { value: v }, v)));
  const opt = (arr) => [h("option", { value: "" }, t("keep")), ...arr.map((v) => h("option", { value: v }, v))];
  const assertion = h("select", { id: "rv-assertion" }, opt(["AFFIRMED", "NEGATED", "UNCERTAIN", "HYPOTHETICAL", "CONDITIONAL", "FAMILY_HISTORY"]));
  const temporal = h("select", { id: "rv-temporal" }, opt(["CURRENT", "HISTORICAL", "PLANNED", "RESOLVED", "UNKNOWN"]));
  const rationale = h("textarea", { id: "rv-rationale", required: true, style: "min-height:70px" }, "");
  const msg = h("div", {});
  const form = h("form", { class: "card", "aria-label": t("review"), onsubmit: async (ev) => {
    ev.preventDefault();
    try {
      await api(`/api/v1/entities/${encodeURIComponent(entity.entity_id)}/review`, { method: "POST", body: {
        review_decision: decision.value,
        corrected_assertion_status: assertion.value || null,
        corrected_temporal_status: temporal.value || null,
        rationale: rationale.value || "Reviewer decision" } });
      toast(t("reviewSaved")); go("document", docId);
    } catch (e) { msg.replaceChildren(alertBox("error", e.message)); }
  } },
    h("h3", {}, `${t("review")}: ${entity.source_text} (${entity.entity_type})`),
    h("div", { class: "form-grid" },
      h("div", {}, h("label", { for: "rv-decision" }, t("decision")), decision),
      h("div", {}, h("label", { for: "rv-assertion" }, t("correctedAssertion")), assertion),
      h("div", {}, h("label", { for: "rv-temporal" }, t("correctedTemporal")), temporal)),
    h("label", { for: "rv-rationale" }, t("rationale")), rationale, msg,
    h("p", {}, h("button", { class: "btn", type: "submit" }, t("submit")), " ",
      h("button", { class: "btn btn-secondary", type: "button", onclick: () => host.replaceChildren() }, t("cancel"))));
  host.replaceChildren(form); form.scrollIntoView({ block: "center" }); decision.focus();
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] || "");
    r.onerror = () => reject(new Error("Could not read file"));
    r.readAsDataURL(file);
  });
}

async function renderIngest() {
  const f = {
    filename: h("input", { id: "in-filename", value: "note.txt" }),
    text: h("textarea", { id: "in-text", placeholder: "CHIEF COMPLAINT:\nPatient presents with chest pain." }),
    file: h("input", { id: "in-file", type: "file", accept: ".txt,.pdf,.docx,.png,.jpg,.jpeg,.tif,.tiff" }),
    subject: h("input", { id: "in-subject", value: "subj_synth_001" }),
    family: h("input", { id: "in-family", value: "fam_synth_upload" }),
    dataset: h("input", { id: "in-dataset", value: "ds_synth_bilingual_v1" }),
    date: h("input", { id: "in-date", type: "date", value: "2026-06-01" }),
    split: h("select", { id: "in-split" }, ["unassigned", "train", "val", "locked_test"].map((v) => h("option", { value: v }, v))),
    specialty: h("input", { id: "in-specialty", value: "INTERNAL_MEDICINE" }),
    dtype: h("input", { id: "in-dtype", value: "CLINICAL_NOTE" }),
  };
  const out = h("div", {});
  const L = (id, key, el) => h("div", {}, h("label", { for: id }, t(key)), el);
  const submit = h("button", { type: "submit", class: "btn" }, t("ingestBtn"));
  const form = h("form", { class: "card", onsubmit: async (ev) => {
    ev.preventDefault(); submit.disabled = true; out.replaceChildren();
    try {
      const body = { filename: f.filename.value, subject_id: f.subject.value, family_id: f.family.value,
        dataset_id: f.dataset.value, encounter_date_iso: f.date.value, split_assignment: f.split.value,
        specialty: f.specialty.value, document_type: f.dtype.value };
      if (f.file.files[0]) { body.filename = f.file.files[0].name; body.content_base64 = await fileToBase64(f.file.files[0]); }
      else body.text = f.text.value;
      const res = await api("/api/v1/documents/ingest", { method: "POST", body });
      out.replaceChildren(alertBox("ok", `${t("ingested")}: ${res.document_id} — ${res.entities_extracted} entities, ${res.relations_extracted} relations`),
        h("button", { type: "button", class: "btn", onclick: () => go("document", res.document_id) }, t("open")));
    } catch (e) { out.replaceChildren(alertBox("error", e.message)); }
    finally { submit.disabled = false; }
  } },
    h("div", { class: "form-grid" }, L("in-filename", "filename", f.filename), L("in-subject", "subjectId", f.subject),
      L("in-family", "familyId", f.family), L("in-dataset", "datasetId", f.dataset), L("in-date", "encounterDate", f.date),
      L("in-split", "split", f.split), L("in-specialty", "specialty", f.specialty), L("in-dtype", "docType", f.dtype)),
    h("label", { for: "in-text" }, t("text")), f.text,
    h("label", { for: "in-file" }, t("orFile")), f.file,
    h("p", {}, submit), out);
  mount(h("h2", {}, t("ingest")), form);
}

async function renderTimeline() {
  const docs = await api("/api/v1/documents");
  const subjects = [...new Set(docs.documents.map((d) => d.subject_id))];
  const select = h("select", { id: "tl-subject" }, subjects.map((s) => h("option", { value: s }, s)));
  const out = h("div", {});
  const load = async () => {
    out.replaceChildren(h("p", { class: "muted" }, t("loading")));
    try {
      const r = await api(`/api/v1/subjects/${encodeURIComponent(select.value)}/timeline`);
      const rows = r.timeline_events.map((e) => h("tr", {},
        h("td", {}, e.event_date_iso), h("td", {}, e.event_category), h("td", {}, e.summary_text),
        h("td", {}, badge(e.assertion_status)), h("td", {}, e.temporal_status), h("td", {}, fmt(e.confidence)),
        h("td", { title: e.contradiction_note || "" }, e.contradiction_flag ? badge("CONTRADICTION", "bad") : "")));
      out.replaceChildren(
        h("p", {}, `${t("contradictions")}: ${r.contradictions_count} · Pseudonym: ${r.subject.pseudonym_code}`),
        h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, [t("date"), t("category"), t("summary"), t("assertion"), t("temporal"), t("conf"), t("flag")].map((c) => h("th", { scope: "col" }, c)))), h("tbody", {}, rows))),
        h("p", { class: "muted" }, "Hover a flag for details. Extracted events are not verified clinical facts."));
    } catch (e) { out.replaceChildren(alertBox("error", e.message)); }
  };
  select.addEventListener("change", load);
  mount(h("h2", {}, t("timeline")), h("div", { class: "card" }, h("label", { for: "tl-subject" }, t("subject")), select, out));
  if (subjects.length) load(); else out.replaceChildren(h("p", {}, t("noData")));
}

async function renderSearch() {
  const q = h("input", { id: "sr-q", value: "hypertension metformin", required: true });
  const out = h("div", {});
  const form = h("form", { class: "card inline-form", onsubmit: async (ev) => {
    ev.preventDefault(); out.replaceChildren(h("p", { class: "muted" }, t("loading")));
    try {
      const r = await api(`/api/v1/search?q=${encodeURIComponent(q.value)}&top_k=10`);
      out.replaceChildren(...(r.results.length ? r.results.map((x) => h("article", { class: "card" },
        h("h3", {}, h("a", { href: "#", onclick: (e) => { e.preventDefault(); go("document", x.document_id); } }, x.original_filename)),
        h("p", {}, `${t("score")}: ${x.hybrid_score} · ${x.detected_language} · ${x.specialty}`),
        h("p", { class: "muted" }, x.preview),
        h("details", {}, h("summary", {}, t("explanation")), h("pre", { class: "json" }, JSON.stringify(x.ranking_explanation, null, 2))))) : [h("p", {}, t("noData"))]));
    } catch (e) { out.replaceChildren(alertBox("error", e.message)); }
  } }, h("div", {}, h("label", { for: "sr-q" }, t("query")), q), h("button", { class: "btn", type: "submit" }, t("run")));
  mount(h("h2", {}, t("search")), form, out);
}

async function renderTerminology() {
  const term = h("input", { id: "tm-term", value: "diabète de type 2", required: true });
  const cat = h("select", { id: "tm-cat" }, [["", t("anyCategory")], ...["DIAGNOSIS", "SYMPTOM", "MEDICATION", "LAB_TEST", "PROCEDURE", "ANATOMY"].map((c) => [c, c])]
    .map(([v, l]) => h("option", { value: v }, l)));
  const out = h("div", {});
  const form = h("form", { class: "card inline-form", onsubmit: async (ev) => {
    ev.preventDefault();
    try {
      const r = await api(`/api/v1/terminology/map?term=${encodeURIComponent(term.value)}${cat.value ? `&category=${cat.value}` : ""}`);
      out.replaceChildren(h("div", { class: "card" },
        h("p", {}, badge(r.match_type, r.match_type === "UNMAPPED" || r.match_type === "AMBIGUOUS" ? "warn" : "gt"), ` ${r.normalized_text}`),
        h("p", {}, r.code ? `${r.vocabulary}: ${r.code} (${t("conf")} ${r.confidence})` : "—"),
        r.candidate_codes.length > 1 ? h("p", { class: "muted" }, "Candidates: " + r.candidate_codes.join(", ")) : null,
        h("p", { class: "muted" }, t("disclaimer"))));
    } catch (e) { out.replaceChildren(alertBox("error", e.message)); }
  } }, h("div", {}, h("label", { for: "tm-term" }, t("term")), term), h("div", {}, h("label", { for: "tm-cat" }, t("category")), cat),
    h("button", { class: "btn", type: "submit" }, t("mapTerm")));
  mount(h("h2", {}, t("terminology")), form, out);
}

async function renderModels() {
  const r = await api("/api/v1/models");
  const rows = r.models.map((m) => h("tr", {},
    h("td", {}, m.model_id), h("td", {}, m.approach_category), h("td", {}, m.track), h("td", {}, m.parameter_count.toLocaleString()),
    h("td", {}, m.init_method), h("td", {}, m.pretrained_weights_used ? badge("PRETRAINED", "bad") : badge("RANDOM INIT", "gt")),
    h("td", {}, badge(m.lifecycle_status, "warn")),
    h("td", { title: m.initial_weights_sha256 }, m.initial_weights_sha256.slice(0, 12) + "…")));
  mount(h("h2", {}, t("models")), h("div", { class: "card table-wrap" },
    h("table", {}, h("thead", {}, h("tr", {}, [t("id"), t("category"), "Track", "Params", "Init", "Weights", "Status", "Init SHA-256"].map((c) => h("th", { scope: "col" }, c)))), h("tbody", {}, rows))),
    h("p", { class: "muted" }, "Registration is not approval. Research metrics do not imply clinical readiness."));
}

function flatten(obj, prefix = "", rows = []) {
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) flatten(v, key, rows);
    else if (typeof v === "number") rows.push([key, v]);
  }
  return rows;
}
function metricsTable(metrics) {
  const rows = flatten(metrics).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, fmt(v, 4))));
  return h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Metric"), h("th", { scope: "col" }, "Value"))), h("tbody", {}, rows)));
}

async function renderExperiments() {
  const r = await api("/api/v1/experiments");
  const out = h("div", {});
  const run = h("button", { type: "button", class: "btn", disabled: !can("training:execute"), onclick: async () => {
    if (!confirm(t("confirmRun"))) return;
    run.disabled = true; out.replaceChildren(h("p", { class: "muted" }, t("runningExp")));
    try {
      const m = await api("/api/v1/experiments/run", { method: "POST" });
      out.replaceChildren(alertBox("ok", `${m.experiment_id} · ${m.resource_usage.runtime_seconds}s · ${m.resource_usage.peak_rss_mb} MB`),
        metricsTable(m.metrics), h("p", { class: "muted" }, m.scientific_limitations.join(" ")));
    } catch (e) { out.replaceChildren(alertBox("error", e.message)); }
    finally { run.disabled = !can("training:execute"); }
  } }, t("runExperiment"));
  const benchOut = h("div", {});
  const runBench = h("button", { type: "button", class: "btn btn-secondary", disabled: !can("training:execute"), onclick: async () => {
    runBench.disabled = true; benchOut.replaceChildren(h("p", { class: "muted" }, t("runningExp")));
    try {
      const b = await api("/api/v1/extraction/benchmark", { method: "POST", body: { epochs: 15, seed: 42 } });
      const rows = b.approaches.map((a) => h("tr", {},
        h("td", {}, a.model_name),
        h("td", {}, String(a.precision)),
        h("td", {}, String(a.recall)),
        h("td", {}, String(a.f1)),
        h("td", {}, String(a.en_f1)),
        h("td", {}, String(a.fr_f1)),
        h("td", {}, String(a.parameter_count)),
        h("td", {}, `${a.train_seconds}s`),
        h("td", {}, `${a.inference_ms_per_doc}ms`)));
      const tbl = h("table", { class: "table" },
        h("thead", {}, h("tr", {},
          h("th", {}, "Model"), h("th", {}, "Prec"), h("th", {}, "Rec"), h("th", {}, "F1"),
          h("th", {}, "EN F1"), h("th", {}, "FR F1"), h("th", {}, "Params"),
          h("th", {}, "Train"), h("th", {}, "Infer/doc"))),
        h("tbody", {}, ...rows));
      benchOut.replaceChildren(
        alertBox("ok", `${b.corpus} · train=${b.train_documents} · test=${b.test_documents} · leakage_verified=${b.leakage_verified} · pretrained=${b.pretrained_weights_used}`),
        tbl);
    } catch (e) { benchOut.replaceChildren(alertBox("error", e.message)); }
    finally { runBench.disabled = !can("training:execute"); }
  } }, "Run 6-Approach Benchmark");
  const dermOut = h("div", {});
  const runDerm = h("button", { type: "button", class: "btn btn-secondary", disabled: !can("training:execute"), onclick: async () => {
    runDerm.disabled = true; dermOut.replaceChildren(h("p", { class: "muted" }, t("runningExp")));
    try {
      const d = await api("/api/v1/dermatology/benchmark", { method: "POST", body: { epochs: 15, per_class_limit: 150, seed: 42 } });
      const toneRows = Object.entries(d.evaluation.by_skin_tone).map(([k, v]) => h("tr", {},
        h("td", {}, k), h("td", {}, String(v.count)), h("td", {}, String(v.accuracy))));
      const toneTbl = h("table", { class: "table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Skin Tone"), h("th", {}, "Count"), h("th", {}, "Acc"))),
        h("tbody", {}, ...toneRows));
      dermOut.replaceChildren(
        alertBox("ok", `DermCV (${d.data_source}) · train=${d.train_samples} · test=${d.test_samples} · acc=${d.evaluation.accuracy} · brier=${d.evaluation.brier_score} · loss ${d.training.initial_loss}→${d.training.final_loss}`),
        toneTbl,
        h("p", { class: "muted" }, d.evaluation.disclaimer));
    } catch (e) { dermOut.replaceChildren(alertBox("error", e.message)); }
    finally { runDerm.disabled = !can("training:execute"); }
  } }, "Run Derm CV Benchmark");
  const cards = r.experiments.map((e) => h("article", { class: "card" },
    h("h3", {}, e.experiment_id),
    h("p", { class: "muted" }, `${e.status} · seed ${e.random_seed} · ${e.runtime_seconds}s · ${e.peak_rss_mb} MB · manifest SHA-256 ${e.manifest_sha256.slice(0, 16)}…`),
    h("details", {}, h("summary", {}, t("metrics")), metricsTable(JSON.parse(e.metrics_json))),
    h("details", {}, h("summary", {}, "Calibration"), metricsTable(JSON.parse(e.calibration_json))),
    h("details", {}, h("summary", {}, "Fairness"), metricsTable(JSON.parse(e.fairness_json)))));
  const rounds = r.federated_rounds.map((x) => h("p", {}, `Federated round ${x.round_number} · ${x.aggregation_algorithm} · sites: ${JSON.parse(x.participating_institutions_json).join(", ")} · ε ≈ ${x.dp_epsilon_spent}`));
  mount(h("h2", {}, t("experiments")),
    h("div", { class: "card" }, run, " ", runBench, " ", runDerm, can("training:execute") ? null : h("p", { class: "muted" }, t("noPermission")), out, benchOut, dermOut),
    ...(cards.length ? cards : [h("p", {}, t("noData"))]),
    rounds.length ? h("section", { class: "card" }, h("h3", {}, "Federated"), rounds) : null,
    h("p", { class: "muted" }, "Experimentally validated on a synthetic benchmark only — not clinically validated."));
}

async function renderAudit() {
  const out = h("div", {});
  const verify = async () => {
    try {
      const r = await api("/api/v1/audit/verify");
      out.replaceChildren(alertBox(r.valid ? "ok" : "error",
        r.valid ? `${t("auditValid")}: ${t("valid")} — ${r.checked_entries} ${t("entries")}. Head: ${r.head_hash_sha256}` : `${t("invalid")}: ${r.error}`));
    } catch (e) { out.replaceChildren(alertBox("error", e.message)); }
  };
  mount(h("h2", {}, t("audit")), h("div", { class: "card" }, h("button", { type: "button", class: "btn", onclick: verify }, t("verifyAudit")), out));
  verify();
}

/* ---------- Login & boot ---------- */
function renderLogin() {
  const user = h("input", { id: "lg-user", autocomplete: "username", required: true });
  const pass = h("input", { id: "lg-pass", type: "password", autocomplete: "current-password", required: true });
  const msg = h("div", {});
  const form = h("form", { class: "card login", onsubmit: async (ev) => {
    ev.preventDefault();
    try {
      const r = await api("/api/v1/auth/token", { method: "POST", body: { username: user.value, password: pass.value } });
      state.token = r.access_token; state.user = { username: user.value, role: r.role, institution_id: r.institution_id, permissions: r.permissions };
      saveSession(); boot();
    } catch (e) { msg.replaceChildren(alertBox("error", `${t("loginFailed")}: ${e.message}`)); pass.value = ""; }
  } },
    h("h2", {}, t("loginTitle")),
    h("label", { for: "lg-user" }, t("username")), user,
    h("label", { for: "lg-pass" }, t("password")), pass, msg,
    h("p", {}, h("button", { class: "btn", type: "submit" }, t("signIn"))));
  document.getElementById("nav").hidden = true;
  mount(form); user.focus();
}

function applyLang() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.getElementById("lang-select").value = state.lang;
}

function boot() {
  applyLang();
  const badgeEl = document.getElementById("user-badge");
  const out = document.getElementById("logout-btn");
  if (!state.token) { badgeEl.hidden = true; out.hidden = true; renderLogin(); return; }
  badgeEl.hidden = false; badgeEl.textContent = `${state.user.username} · ${state.user.role} · ${state.user.institution_id}`;
  out.hidden = false;
  go(state.view === "document" ? "documents" : state.view);
}

document.getElementById("lang-select").addEventListener("change", (e) => {
  state.lang = e.target.value; localStorage.setItem("lang", state.lang); boot();
});
document.getElementById("logout-btn").addEventListener("click", logout);
loadSession();
boot();
