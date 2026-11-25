import { useState, type DragEvent, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, UploadCloud } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest, uploadDocumentMultipart, uploadImageMultipart } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { JobStatus } from "@/types/api";

const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "tif", "tiff"]);

const QUICK_TEXT_PRESETS: Array<{ id: string; labelEn: string; labelFr: string; filename: string; text: string }> = [
  {
    id: "fr_vitals",
    labelEn: "FR Vitals & Meds",
    labelFr: "FR Constantes & Traitement",
    filename: "note_fr_constantes.txt",
    text: "Le patient a une tension artérielle de 140/90 mmHg et un poids de 82 kg. Pas de fièvre. Sa mère a un diabète de type 2. Début de metformine 500 mg 2x/j PO pendant 14 jours.",
  },
  {
    id: "en_note",
    labelEn: "EN Clinical Note",
    labelFr: "EN Note Clinique",
    filename: "note_en_clinical.txt",
    text: "Started metformin 500 mg BID PO for 14 days for type 2 diabetes, and lisinopril 10 mg QD PO for hypertension. No fever. History of asthma. Mother has diabetes.",
  },
  {
    id: "fr_dialogue",
    labelEn: "FR Doctor–Patient",
    labelFr: "FR Dialogue Médecin–Patient",
    filename: "dialogue_fr_consultation.txt",
    text: "Médecin : Bonjour, qu'est-ce qui vous amène ?\nPatient : J'ai de la toux mais pas de fièvre. Ma mère a un diabète.\nMédecin : Votre HbA1c est à 7.4 %. Débutez amoxicilline 1 g 3x/j PO pendant 7 jours.",
  },
];

function isImageFile(file: File): boolean {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.has(ext) || file.type.startsWith("image/");
}

export function UploadJobsPage() {
  const { t, i18n } = useTranslation();
  const token = useAuthStore((s) => s.token);
  const queryClient = useQueryClient();

  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [file, setFile] = useState<File | null>(null);
  const [customText, setCustomText] = useState("");
  const [customFilename, setCustomFilename] = useState("custom_clinical_note.txt");
  const [subjectId, setSubjectId] = useState("subj_synth_001");
  const [specialty, setSpecialty] = useState("INTERNAL_MEDICINE");
  const [bodySite, setBodySite] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [textSubmittedNotice, setTextSubmittedNotice] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [triageResult, setTriageResult] = useState<Record<string, unknown> | null>(null);

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => apiRequest<{ jobs: JobStatus[]; count: number }>("/api/v1/jobs", { token }),
    enabled: Boolean(token),
    refetchInterval: 2000,
  });

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: () => apiRequest<{ vision_router?: string; vision_skin?: string }>("/api/v1/health", { token }),
  });
  const visionModelsMissing =
    healthQuery.data?.vision_router === "missing" || healthQuery.data?.vision_skin === "missing";

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0] ?? null;
    if (dropped) {
      setFile(dropped);
      setTriageResult(null);
      setError(null);
    }
  };

  const handleUpload = async (e: FormEvent) => {
    e.preventDefault();
    if (!token) return;
    setError(null);
    setTriageResult(null);
    setTextSubmittedNotice(null);

    if (inputMode === "text") {
      const trimmed = customText.trim();
      if (!trimmed) return;
      const safeBase = (customFilename.trim() || "custom_clinical_note.txt").replace(/[^a-zA-Z0-9._-]/g, "_");
      const finalName = safeBase.toLowerCase().endsWith(".txt") ? safeBase : `${safeBase}.txt`;
      const textFile = new File([trimmed], finalName, { type: "text/plain;charset=utf-8" });
      setSubmitting(true);
      try {
        await uploadDocumentMultipart(textFile, token, {
          subject_id: subjectId,
          specialty,
          document_type: "CLINICAL_NOTE",
        });
        setTextSubmittedNotice(finalName);
        void queryClient.invalidateQueries({ queryKey: ["jobs"] });
        void queryClient.invalidateQueries({ queryKey: ["documents"] });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Text submission failed");
      } finally {
        setSubmitting(false);
      }
      return;
    }

    if (!file) return;
    setSubmitting(true);
    try {
      if (isImageFile(file)) {
        const res = await uploadImageMultipart(file, token, {
          subject_id: subjectId,
          body_site: bodySite.trim() || undefined,
        });
        setTriageResult(res);
        void queryClient.invalidateQueries({ queryKey: ["images"] });
        void queryClient.invalidateQueries({ queryKey: ["patient-record"] });
      } else {
        await uploadDocumentMultipart(file, token, {
          subject_id: subjectId,
          specialty,
          document_type: "CLINICAL_NOTE",
        });
      }
      setFile(null);
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  };

  const triageObj = (triageResult?.triage as Record<string, unknown> | undefined) ?? undefined;
  const triageLabel =
    i18n.language === "fr"
      ? (triageObj?.label_fr as string | undefined)
      : (triageObj?.label_en as string | undefined);

  return (
    <div className="space-y-6">
      {visionModelsMissing && (
        <div
          role="alert"
          className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200"
        >
          <p className="font-semibold">
            {i18n.language === "fr"
              ? "Modèle image non entraîné — l'envoi d'images renverra 503."
              : "Image model not trained — image uploads will return 503."}
          </p>
          <p className="mt-1 font-mono">medextract derm train --seed 1 --out models/derm_cnn_seed1</p>
        </div>
      )}
      <div
        role="note"
        className="rounded-lg border border-sky-200 bg-sky-50/70 px-3 py-2 text-xs text-slate-700 dark:border-sky-900 dark:bg-sky-950/30 dark:text-slate-200"
      >
        {t("upload.servedCapabilityNotice")}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            {inputMode === "file" ? (
              <UploadCloud className="h-5 w-5 text-sky-700 dark:text-sky-400" aria-hidden="true" />
            ) : (
              <FileText className="h-5 w-5 text-sky-700 dark:text-sky-400" aria-hidden="true" />
            )}
            <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{t("upload.title")}</h1>
          </div>
          <div
            role="tablist"
            aria-label={i18n.language === "fr" ? "Mode d'entrée" : "Input mode"}
            className="inline-flex rounded-md border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-900 p-0.5 text-xs"
          >
            <button
              type="button"
              role="tab"
              aria-selected={inputMode === "file"}
              onClick={() => {
                setInputMode("file");
                setError(null);
              }}
              className={`rounded px-2.5 py-1 font-medium transition-colors ${
                inputMode === "file"
                  ? "bg-white dark:bg-slate-800 text-sky-700 dark:text-sky-300 shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
              }`}
            >
              {i18n.language === "fr" ? "Fichier / Image" : "Upload File"}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={inputMode === "text"}
              onClick={() => {
                setInputMode("text");
                setError(null);
              }}
              className={`rounded px-2.5 py-1 font-medium transition-colors ${
                inputMode === "text"
                  ? "bg-white dark:bg-slate-800 text-sky-700 dark:text-sky-300 shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
              }`}
            >
              {i18n.language === "fr" ? "Écrire / Coller un texte" : "Write / Paste Text"}
            </button>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={(e) => void handleUpload(e)} className="space-y-4">
            {inputMode === "file" ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                className={`rounded-lg border-2 border-dashed p-4 text-center transition-colors ${
                  dragActive
                    ? "border-sky-500 bg-sky-50/70 dark:bg-sky-950/40"
                    : "border-slate-300 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-900/40"
                }`}
              >
                <label htmlFor="doc-file" className="block text-xs font-medium text-slate-700 dark:text-slate-200 mb-2">
                  {t("upload.dropHint")}
                </label>
                <input
                  id="doc-file"
                  type="file"
                  accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.tiff,.webp"
                  onChange={(e) => {
                    setFile(e.target.files?.[0] ?? null);
                    setTriageResult(null);
                  }}
                  className="block w-full text-xs text-slate-700 dark:text-slate-300 file:mr-3 file:rounded file:border-0 file:bg-sky-100 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-sky-800 dark:file:bg-sky-900 dark:file:text-sky-200"
                />
                {file && (
                  <p className="mt-2 text-xs text-slate-600 dark:text-slate-300">
                    Selected: <span className="font-mono font-medium">{file.name}</span> (
                    {isImageFile(file) ? "Image → Triage Router" : "Document → Text/OCR Pipeline"})
                  </p>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label htmlFor="custom-clinical-text" className="text-xs font-medium text-slate-700 dark:text-slate-200">
                    {i18n.language === "fr"
                      ? "Texte clinique ou dialogue médecin–patient (EN / FR)"
                      : "Clinical note or doctor–patient dialogue (EN / FR)"}
                  </label>
                  <div className="flex flex-wrap gap-1.5">
                    {QUICK_TEXT_PRESETS.map((preset) => (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => {
                          setCustomText(preset.text);
                          setCustomFilename(preset.filename);
                          setTextSubmittedNotice(null);
                          setError(null);
                        }}
                        className="rounded border border-sky-200 dark:border-sky-800 bg-sky-50 dark:bg-sky-950/40 px-2 py-0.5 text-[11px] font-medium text-sky-800 dark:text-sky-200 hover:bg-sky-100 dark:hover:bg-sky-900/60"
                      >
                        {i18n.language === "fr" ? preset.labelFr : preset.labelEn}
                      </button>
                    ))}
                  </div>
                </div>
                <textarea
                  id="custom-clinical-text"
                  rows={6}
                  value={customText}
                  onChange={(e) => {
                    setCustomText(e.target.value);
                    setTextSubmittedNotice(null);
                  }}
                  placeholder={
                    i18n.language === "fr"
                      ? "Ex. : Le patient a une tension artérielle de 140/90 mmHg et un poids de 82 kg. Pas de fièvre. Sa mère a un diabète..."
                      : "e.g., Patient started metformin 500 mg BID PO for 14 days for type 2 diabetes. No fever. Mother has hypertension..."
                  }
                  className="w-full rounded-lg border border-slate-300 dark:border-slate-700 dark:bg-slate-900 p-3 text-xs leading-relaxed text-slate-900 dark:text-slate-100 focus:border-sky-500 focus:outline-none"
                />
                <div className="flex items-center justify-between gap-3">
                  <div className="flex-1">
                    <label htmlFor="custom-filename" className="block text-[11px] font-medium text-slate-600 dark:text-slate-400 mb-1">
                      {i18n.language === "fr" ? "Nom du document (.txt)" : "Document name (.txt)"}
                    </label>
                    <input
                      id="custom-filename"
                      type="text"
                      value={customFilename}
                      onChange={(e) => setCustomFilename(e.target.value)}
                      className="w-full rounded border border-slate-300 dark:border-slate-700 dark:bg-slate-900 px-2.5 py-1 text-xs font-mono"
                    />
                  </div>
                  <span className="self-end pb-1 text-[11px] text-slate-500">
                    {customText.length} {i18n.language === "fr" ? "caractères" : "chars"}
                  </span>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label htmlFor="subject-id" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Subject ID
                </label>
                <input
                  id="subject-id"
                  type="text"
                  value={subjectId}
                  onChange={(e) => setSubjectId(e.target.value)}
                  className="w-full rounded border border-slate-300 dark:border-slate-700 dark:bg-slate-900 px-2.5 py-1.5 text-xs"
                />
              </div>
              <div>
                <label htmlFor="specialty" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Specialty
                </label>
                <input
                  id="specialty"
                  type="text"
                  value={specialty}
                  onChange={(e) => setSpecialty(e.target.value)}
                  className="w-full rounded border border-slate-300 dark:border-slate-700 dark:bg-slate-900 px-2.5 py-1.5 text-xs"
                />
              </div>
              <div>
                <label htmlFor="body-site" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                  {t("upload.bodySiteLabel")}
                </label>
                <input
                  id="body-site"
                  type="text"
                  placeholder="e.g. left forearm / dos"
                  value={bodySite}
                  onChange={(e) => setBodySite(e.target.value)}
                  className="w-full rounded border border-slate-300 dark:border-slate-700 dark:bg-slate-900 px-2.5 py-1.5 text-xs"
                />
              </div>
            </div>

            {error && (
              <div role="alert" className="rounded bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-800 px-3 py-2 text-xs text-rose-800 dark:text-rose-200">
                {error}
              </div>
            )}

            {triageResult && (
              <div
                data-testid="triage-result-banner"
                role="status"
                className="rounded-md bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 p-3 text-xs space-y-1.5"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-emerald-900 dark:text-emerald-200">
                    {t("upload.triageDetected")}: {triageLabel ?? String(triageObj?.predicted_class ?? "skin_image")}
                  </span>
                  <Badge tone="emerald">{String(triageResult.routed_to ?? "skin_vision_pipeline")}</Badge>
                </div>
                {typeof triageResult.image_id === "string" && (
                  <p className="text-slate-700 dark:text-slate-300">
                    Image ID: <span className="font-mono">{triageResult.image_id}</span> ·{" "}
                    <Link to="/skin" className="underline font-medium text-sky-700 dark:text-sky-300">
                      Open in Skin & Patient Workspace →
                    </Link>
                  </p>
                )}
              </div>
            )}

            {textSubmittedNotice && (
              <div
                data-testid="text-submitted-banner"
                role="status"
                className="rounded-md bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 p-3 text-xs flex flex-wrap items-center justify-between gap-2"
              >
                <span className="font-medium text-emerald-900 dark:text-emerald-200">
                  {i18n.language === "fr"
                    ? `Texte soumis pour extraction : ${textSubmittedNotice}`
                    : `Text submitted for extraction: ${textSubmittedNotice}`}
                </span>
                <Link to="/workspace" className="underline font-semibold text-sky-700 dark:text-sky-300">
                  {i18n.language === "fr" ? "Ouvrir dans l'Espace de travail →" : "Open in Workspace →"}
                </Link>
              </div>
            )}

            <Button
              type="submit"
              disabled={(inputMode === "file" ? !file : !customText.trim()) || submitting}
            >
              {submitting
                ? "..."
                : inputMode === "text"
                ? i18n.language === "fr"
                  ? "Lancer l'extraction sur le texte"
                  : "Run Extraction on Text"
                : t("upload.submit")}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{t("upload.jobsTitle")}</h2>
        </CardHeader>
        <CardContent className="space-y-3 max-h-[65vh] overflow-y-auto">
          {jobsQuery.data?.jobs.length === 0 && (
            <p className="text-xs text-slate-500">No background jobs yet.</p>
          )}
          {jobsQuery.data?.jobs.map((job) => (
            <div key={job.job_id} className="rounded-md border border-slate-200 dark:border-slate-800 p-3 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono font-medium">{job.job_id}</span>
                <Badge
                  tone={
                    job.status === "SUCCEEDED"
                      ? "emerald"
                      : job.status === "FAILED"
                      ? "rose"
                      : "sky"
                  }
                >
                  {job.status} · {job.current_stage}
                </Badge>
              </div>
              <div className="w-full h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-sky-600 transition-all"
                  style={{ width: `${job.progress_pct}%` }}
                />
              </div>
              {job.error_message && <p className="text-rose-700 dark:text-rose-300">{job.error_message}</p>}
            </div>
          ))}
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
