import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest, uploadDocumentMultipart } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { JobStatus } from "@/types/api";

export function UploadJobsPage() {
  const { t } = useTranslation();
  const token = useAuthStore((s) => s.token);
  const queryClient = useQueryClient();

  const [file, setFile] = useState<File | null>(null);
  const [subjectId, setSubjectId] = useState("subj_synth_001");
  const [specialty, setSpecialty] = useState("INTERNAL_MEDICINE");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => apiRequest<{ jobs: JobStatus[]; count: number }>("/api/v1/jobs", { token }),
    enabled: Boolean(token),
    refetchInterval: 2000,
  });

  const handleUpload = async (e: FormEvent) => {
    e.preventDefault();
    if (!file || !token) return;
    setError(null);
    setSubmitting(true);
    try {
      await uploadDocumentMultipart(file, token, {
        subject_id: subjectId,
        specialty,
        document_type: "CLINICAL_NOTE",
      });
      setFile(null);
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader className="flex items-center gap-2">
          <UploadCloud className="h-5 w-5 text-sky-700" aria-hidden="true" />
          <h1 className="text-sm font-semibold text-slate-900">{t("upload.title")}</h1>
        </CardHeader>
        <CardContent>
          <form onSubmit={(e) => void handleUpload(e)} className="space-y-4">
            <div>
              <label htmlFor="doc-file" className="block text-xs font-medium text-slate-700 mb-1">
                Document File
              </label>
              <input
                id="doc-file"
                type="file"
                accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.tiff,.webp"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="block w-full text-xs text-slate-700 file:mr-3 file:rounded file:border-0 file:bg-sky-50 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-sky-800"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="subject-id" className="block text-xs font-medium text-slate-700 mb-1">
                  Subject ID
                </label>
                <input
                  id="subject-id"
                  type="text"
                  value={subjectId}
                  onChange={(e) => setSubjectId(e.target.value)}
                  className="w-full rounded border border-slate-300 px-2.5 py-1.5 text-xs"
                />
              </div>
              <div>
                <label htmlFor="specialty" className="block text-xs font-medium text-slate-700 mb-1">
                  Specialty
                </label>
                <input
                  id="specialty"
                  type="text"
                  value={specialty}
                  onChange={(e) => setSpecialty(e.target.value)}
                  className="w-full rounded border border-slate-300 px-2.5 py-1.5 text-xs"
                />
              </div>
            </div>

            {error && (
              <div role="alert" className="rounded bg-rose-50 border border-rose-200 px-3 py-2 text-xs text-rose-800">
                {error}
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
          <h2 className="text-sm font-semibold text-slate-900">{t("upload.jobsTitle")}</h2>
        </CardHeader>
        <CardContent className="space-y-3 max-h-[65vh] overflow-y-auto">
          {jobsQuery.data?.jobs.length === 0 && (
            <p className="text-xs text-slate-500">No background jobs yet.</p>
          )}
          {jobsQuery.data?.jobs.map((job) => (
            <div key={job.job_id} className="rounded-md border border-slate-200 p-3 text-xs space-y-2">
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
              <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full bg-sky-600 transition-all"
                  style={{ width: `${job.progress_pct}%` }}
                />
              </div>
              {job.error_message && <p className="text-rose-700">{job.error_message}</p>}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
