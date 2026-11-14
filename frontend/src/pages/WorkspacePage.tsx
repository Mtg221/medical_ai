import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, FileSearch, Network, Code2, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { DocumentDetailResponse, DocumentSummary, ExtractedEntity } from "@/types/api";

function HighlightedPageText({
  text,
  entities,
  selectedId,
  onSelect,
}: {
  text: string;
  entities: ExtractedEntity[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (!text) return <p className="text-xs text-slate-400 italic">Empty page</p>;

  const valid = [...entities]
    .filter((e) => e.char_start >= 0 && e.char_end > e.char_start && e.char_end <= text.length)
    .sort((a, b) => a.char_start - b.char_start);

  const nodes: React.ReactNode[] = [];
  let cursor = 0;

  for (const ent of valid) {
    if (ent.char_start < cursor) continue;
    if (ent.char_start > cursor) {
      nodes.push(<span key={`t-${cursor}`}>{text.slice(cursor, ent.char_start)}</span>);
    }
    const isSelected = ent.entity_id === selectedId;
    const isNegated = ent.assertion_status === "absent";
    nodes.push(
      <button
        key={ent.entity_id}
        type="button"
        onClick={() => onSelect(ent.entity_id)}
        className={`inline rounded px-1 py-0.5 text-left font-medium transition-colors cursor-pointer ${
          isSelected
            ? "bg-sky-600 text-white ring-2 ring-sky-300"
            : isNegated
            ? "bg-rose-100 text-rose-900 border-b-2 border-rose-400"
            : "bg-amber-100 text-amber-950 border-b-2 border-amber-400 hover:bg-amber-200"
        }`}
        title={`${ent.entity_type} · ${ent.assertion_status}`}
      >
        {text.slice(ent.char_start, ent.char_end)}
      </button>
    );
    cursor = ent.char_end;
  }
  if (cursor < text.length) {
    nodes.push(<span key={`t-end`}>{text.slice(cursor)}</span>);
  }

  return <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-800">{nodes}</pre>;
}

export function WorkspacePage() {
  const { t } = useTranslation();
  const token = useAuthStore((s) => s.token);
  const queryClient = useQueryClient();
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<"entities" | "graph" | "json">("entities");

  const docsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: () =>
      apiRequest<{ documents: DocumentSummary[]; count: number }>("/api/v1/documents", { token }),
    enabled: Boolean(token),
  });

  const activeDocId = selectedDocId ?? docsQuery.data?.documents[0]?.document_id ?? null;

  const detailQuery = useQuery({
    queryKey: ["document", activeDocId],
    queryFn: () => apiRequest<DocumentDetailResponse>(`/api/v1/documents/${activeDocId}`, { token }),
    enabled: Boolean(token && activeDocId),
  });

  const reviewMutation = useMutation({
    mutationFn: ({ entityId, decision }: { entityId: string; decision: string }) =>
      apiRequest(`/api/v1/entities/${entityId}/review`, {
        method: "POST",
        token,
        body: { review_decision: decision, rationale: `Reviewed via React workspace (${decision})` },
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["document", activeDocId] });
    },
  });

  const detail = detailQuery.data;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
      {/* Document Sidebar */}
      <Card className="lg:col-span-3">
        <CardHeader>
          <h1 className="text-sm font-semibold text-slate-900">{t("workspace.documents")}</h1>
        </CardHeader>
        <CardContent className="p-2 space-y-1 max-h-[70vh] overflow-y-auto">
          {docsQuery.data?.documents.length === 0 && (
            <p className="text-xs text-slate-500 p-2">{t("workspace.noDocs")}</p>
          )}
          {docsQuery.data?.documents.map((doc) => (
            <button
              key={doc.document_id}
              type="button"
              onClick={() => {
                setSelectedDocId(doc.document_id);
                setSelectedEntityId(null);
              }}
              className={`w-full text-left rounded-md p-2.5 text-xs transition-colors cursor-pointer ${
                doc.document_id === activeDocId
                  ? "bg-sky-50 border border-sky-200 text-sky-950"
                  : "hover:bg-slate-100 text-slate-800"
              }`}
            >
              <div className="font-medium truncate">{doc.original_filename}</div>
              <div className="flex flex-wrap gap-1 mt-1">
                <Badge tone="slate">{doc.file_format}</Badge>
                <Badge tone="sky">{doc.detected_language.toUpperCase()}</Badge>
                <Badge tone="emerald">{doc.extraction_mode}</Badge>
              </div>
            </button>
          ))}
        </CardContent>
      </Card>

      {/* Source-Linked Viewer */}
      <Card className="lg:col-span-5">
        <CardHeader className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileSearch className="h-4 w-4 text-sky-700" aria-hidden="true" />
            <h2 className="text-sm font-semibold text-slate-900">{t("workspace.sourceViewer")}</h2>
          </div>
          {detail && <Badge tone="slate">{detail.document.page_count} p.</Badge>}
        </CardHeader>
        <CardContent className="space-y-4 max-h-[70vh] overflow-y-auto">
          {!detail && <p className="text-xs text-slate-500">Select a document to inspect.</p>}
          {detail?.pages.map((p) => (
            <div key={p.page_id} className="rounded-md border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center justify-between text-[11px] text-slate-500 mb-2">
                <span>
                  Page {p.page_number} · {p.extraction_source}
                </span>
                <span>conf {(p.mean_confidence * 100).toFixed(0)}%</span>
              </div>
              <HighlightedPageText
                text={p.raw_text}
                entities={detail.entities.filter((e) => e.page_number === p.page_number)}
                selectedId={selectedEntityId}
                onSelect={setSelectedEntityId}
              />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Right Pane: Entities / Graph / JSON */}
      <Card className="lg:col-span-4">
        <CardHeader className="flex items-center gap-1 p-2">
          <Button
            size="sm"
            variant={rightTab === "entities" ? "primary" : "outline"}
            onClick={() => setRightTab("entities")}
          >
            {t("workspace.entities")}
          </Button>
          <Button
            size="sm"
            variant={rightTab === "graph" ? "primary" : "outline"}
            onClick={() => setRightTab("graph")}
          >
            <Network className="h-3.5 w-3.5" aria-hidden="true" />
            {t("workspace.relations")}
          </Button>
          <Button
            size="sm"
            variant={rightTab === "json" ? "primary" : "outline"}
            onClick={() => setRightTab("json")}
          >
            <Code2 className="h-3.5 w-3.5" aria-hidden="true" />
            {t("workspace.jsonExport")}
          </Button>
        </CardHeader>

        <CardContent className="max-h-[70vh] overflow-y-auto space-y-2.5">
          {rightTab === "entities" &&
            detail?.entities.map((ent) => (
              <div
                key={ent.entity_id}
                onClick={() => setSelectedEntityId(ent.entity_id)}
                className={`rounded-md border p-2.5 text-xs cursor-pointer ${
                  ent.entity_id === selectedEntityId
                    ? "border-sky-400 bg-sky-50"
                    : "border-slate-200 bg-white hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-900">{ent.surface_text}</span>
                  <Badge tone={ent.assertion_status === "absent" ? "rose" : "emerald"}>
                    {ent.assertion_status}
                  </Badge>
                </div>
                <div className="text-[11px] text-slate-600 mt-1">
                  {ent.entity_type} · {ent.normalized_text} · span [{ent.char_start}, {ent.char_end}]
                </div>
                <div className="flex items-center justify-between mt-2">
                  <Badge tone="slate">{ent.review_status}</Badge>
                  <div className="flex gap-1">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        reviewMutation.mutate({ entityId: ent.entity_id, decision: "APPROVE" });
                      }}
                    >
                      <Check className="h-3 w-3 text-emerald-700" aria-hidden="true" />
                      {t("workspace.approve")}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        reviewMutation.mutate({ entityId: ent.entity_id, decision: "REJECT" });
                      }}
                    >
                      <X className="h-3 w-3 text-rose-700" aria-hidden="true" />
                      {t("workspace.reject")}
                    </Button>
                  </div>
                </div>
              </div>
            ))}

          {rightTab === "graph" && detail && (
            <div className="space-y-2">
              {detail.relations.length === 0 ? (
                <p className="text-xs text-slate-500">No relations extracted on this document.</p>
              ) : (
                detail.relations.map((rel) => {
                  const src = detail.entities.find((e) => e.entity_id === rel.source_entity_id);
                  const tgt = detail.entities.find((e) => e.entity_id === rel.target_entity_id);
                  return (
                    <div key={rel.relation_id} className="rounded border border-slate-200 p-2 text-xs">
                      <span className="font-medium">{src?.surface_text ?? rel.source_entity_id}</span>{" "}
                      <Badge tone="sky">{rel.relation_type}</Badge>{" "}
                      <span className="font-medium">{tgt?.surface_text ?? rel.target_entity_id}</span>
                    </div>
                  );
                })
              )}
            </div>
          )}

          {rightTab === "json" && detail && (
            <pre className="text-[11px] font-mono bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto">
              {JSON.stringify(detail, null, 2)}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
