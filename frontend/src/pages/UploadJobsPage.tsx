import { useState, type DragEvent, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest, uploadDocumentMultipart, uploadImageMultipart } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { JobStatus } from "@/types/api";

const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "tif", "tiff"]);

function isImageFile(file: File): boolean {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.has(ext) || file.type.startsWith("image/");
}

export function UploadJobsPage() {
  const { t, i18n } = useTranslation();
  const token = useAuthStore((s) => s.token);
  const queryClient = useQueryClient();

  const [file, setFile] = useState<File | null>(null);
  const [subjectId, setSubjectId] = useState("subj_synth_001");
  const [specialty, setSpecialty] = useState("DERMATOLOGY");
  const [bodySite, setBodySite] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [triageResult, setTriageResult] = useState<Record<string, unknown> | null>(null);

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => apiRequest<{ jobs: JobStatus[]; count: number }>("/api/v1/jobs", { token }),
    enabled: Boolean(token),
    refetchInterval: 2000,
  });

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
    if (!file || !token) return;
    setError(null);
    setTriageResult(null);
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
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader className="flex items-center gap-2">
          <UploadCloud className="h-5 w-5 text-sky-700 dark:text-sky-400" aria-hidden="true" />
          <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{t("upload.title")}</h1>
        </CardHeader>
        <CardContent>
          <form onSubmit={(e) => void handleUpload(e)} className="space-y-4">
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

            <Button type="submit" disabled={!file || submitting}>
              {submitting ? "..." : t("upload.submit")}
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
  );
}
