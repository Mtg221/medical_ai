import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, Flag, X } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { DocumentDetailResponse, DocumentSummary } from "@/types/api";

export function ReviewQueuePage() {
  const token = useAuthStore((s) => s.token);
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [onlyFlagged, setOnlyFlagged] = useState<boolean>(false);
  const [decisions, setDecisions] = useState<Record<string, "approved" | "rejected" | "flagged">>({});

  const docsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: () => apiRequest<{ documents: DocumentSummary[] }>("/api/v1/documents", { token }),
    enabled: Boolean(token),
  });

  const activeDocId = selectedDocId || docsQuery.data?.documents[0]?.document_id || "";

  const detailQuery = useQuery({
    queryKey: ["document-detail", activeDocId],
    queryFn: () => apiRequest<DocumentDetailResponse>(`/api/v1/documents/${activeDocId}`, { token }),
    enabled: Boolean(token && activeDocId),
  });

  const entities = (detailQuery.data?.entities ?? []).filter((e) =>
    onlyFlagged ? e.confidence < 0.75 || e.review_status === "flagged" : true
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-semibold text-slate-900">Human-in-the-Loop Review Mode</h1>
            <p className="text-xs text-slate-500">Verify extracted entities, assertion labels, and low-confidence flags</p>
          </div>
          <div className="flex items-center gap-3">
            <label className="inline-flex items-center gap-1.5 text-xs text-slate-700">
              <input
                type="checkbox"
                checked={onlyFlagged}
                onChange={(e) => setOnlyFlagged(e.target.checked)}
              />
              Low-confidence / Flagged only
            </label>
            <select
              aria-label="Review document"
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
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          {entities.length === 0 ? (
            <p className="text-xs text-slate-500 py-6 text-center">No entities match the current filter.</p>
          ) : (
            entities.map((ent) => {
              const status = decisions[ent.entity_id] ?? ent.review_status;
              return (
                <div
                  key={ent.entity_id}
                  className="rounded border border-slate-200 p-3 flex flex-wrap items-center justify-between gap-3 text-xs"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-slate-900">{ent.surface_text}</span>
                      <Badge tone="sky">{ent.entity_type}</Badge>
                      <Badge tone={ent.assertion_status === "present" ? "emerald" : "amber"}>
                        {ent.assertion_status}
                      </Badge>
                      <span className="text-slate-500">conf {(ent.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="text-slate-500">
                      Offsets [{ent.char_start}, {ent.char_end}] · Status: <span className="font-medium">{status}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setDecisions((p) => ({ ...p, [ent.entity_id]: "approved" }))}
                      aria-label={`Approve ${ent.surface_text}`}
                    >
                      <Check className="h-3.5 w-3.5 text-emerald-700" />
                      Approve
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setDecisions((p) => ({ ...p, [ent.entity_id]: "flagged" }))}
                      aria-label={`Flag ${ent.surface_text}`}
                    >
                      <Flag className="h-3.5 w-3.5 text-amber-600" />
                      Flag
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setDecisions((p) => ({ ...p, [ent.entity_id]: "rejected" }))}
                      aria-label={`Reject ${ent.surface_text}`}
                    >
                      <X className="h-3.5 w-3.5 text-rose-600" />
                      Reject
                    </Button>
                  </div>
                </div>
              );
            })
          )}
        </CardContent>
      </Card>
    </div>
  );
}
