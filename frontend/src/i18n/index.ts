import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const resources = {
  en: {
    translation: {
      appTitle: "MedExtract Research",
      disclaimer:
        "Research tool only — NOT for clinical diagnosis or treatment; human review required before clinical use.",
      nav: {
        workspace: "Document Workspace",
        upload: "Upload & Jobs",
        benchmark: "Models & Benchmark",
        timeline: "Timeline & Audit",
      },
      auth: {
        signInTitle: "Sign in to Research Workspace",
        signInSubtitle: "Institution-scoped authentication (no default credentials stored).",
        username: "Username",
        password: "Password",
        signIn: "Sign In",
        signOut: "Sign Out",
      },
      workspace: {
        documents: "Documents",
        sourceViewer: "Source-Linked Document Viewer",
        entities: "Extracted Facts & Review",
        relations: "Relation Graph",
        jsonExport: "Structured JSON",
        approve: "Approve",
        reject: "Reject",
        noDocs: "No documents found for this institution. Upload one in Upload & Jobs.",
      },
      upload: {
        title: "Upload Clinical Document (PDF, DOCX, TXT, Scan Image)",
        submit: "Start Extraction Job",
        jobsTitle: "Background Extraction Jobs (SQLite + SSE)",
      },
    },
  },
  fr: {
    translation: {
      appTitle: "MedExtract Recherche",
      disclaimer:
        "Outil de recherche uniquement — NON destiné au diagnostic ou au traitement ; révision humaine requise avant tout usage clinique.",
      nav: {
        workspace: "Espace Documents",
        upload: "Import & Tâches",
        benchmark: "Modèles & Évaluation",
        timeline: "Chronologie & Audit",
      },
      auth: {
        signInTitle: "Connexion à l'espace de recherche",
        signInSubtitle: "Authentification par institution (aucun mot de passe par défaut).",
        username: "Nom d'utilisateur",
        password: "Mot de passe",
        signIn: "Se connecter",
        signOut: "Déconnexion",
      },
      workspace: {
        documents: "Documents",
        sourceViewer: "Visionneuse liée à la source",
        entities: "Faits extraits & Révision",
        relations: "Graphe de relations",
        jsonExport: "JSON structuré",
        approve: "Approuver",
        reject: "Rejeter",
        noDocs: "Aucun document pour cette institution. Importez un fichier dans Import & Tâches.",
      },
      upload: {
        title: "Importer un document clinique (PDF, DOCX, TXT, Image scannée)",
        submit: "Lancer l'extraction",
        jobsTitle: "Tâches d'extraction en arrière-plan (SQLite + SSE)",
      },
    },
  },
};

i18n.use(initReactI18next).init({
  resources,
  lng: "en",
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

export default i18n;
