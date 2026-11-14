import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { DocumentSummary } from "@/types/api";

export function ReportsPage() {
  const token = useAuthStore((s) => s.token);
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [lang, setLang] = useState<"en" | "fr">("en");

  const docsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: () => apiRequest<{ documents: DocumentSummary[] }>("/api/v1/documents", { token }),
    enabled: Boolean(token),
  });

  const activeDocId = selectedDocId || docsQuery.data?.documents[0]?.document_id || "";

  const reportQuery = useQuery({
    queryKey: ["report-preview", activeDocId, lang],
    queryFn: () =>
      apiRequest<{ markdown?: string; report_markdown?: string }>(
        `/api/v1/documents/${activeDocId}/report?language=${lang}&format=markdown`,
        { token }
      ),
    enabled: Boolean(token && activeDocId),
  });

  const mdText = reportQuery.data?.markdown ?? reportQuery.data?.report_markdown ?? "";

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-semibold text-slate-900">Traceable Clinical Reports (EN / FR)</h1>
            <p className="text-xs text-slate-500">Every finding cites [doc, page, char_start-char_end] with negation & disclaimer</p>
          </div>
          <div className="flex items-center gap-2">
            <select
              aria-label="Report document"
              value={activeDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
              className="rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs"
            >
              {docsQuery.data?.documents.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  {d.original_filename || d.document_id}
                </option>
              ))}
            </select>
            <Button size="sm" variant={lang === "en" ? "primary" : "outline"} onClick={() => setLang("en")}>
              EN
            </Button>
            <Button size="sm" variant={lang === "fr" ? "primary" : "outline"} onClick={() => setLang("fr")}>
              FR
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {reportQuery.isLoading ? (
            <p className="text-xs text-slate-500">Generating traceable report...</p>
          ) : (
            <pre className="whitespace-pre-wrap rounded border border-slate-200 bg-slate-50 p-4 text-xs font-mono text-slate-800 max-h-[520px] overflow-auto">
              {mdText || "Select an extracted document to preview its traceable EN/FR report."}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
