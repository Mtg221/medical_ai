import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Clock, FileSearch, Network, Code2, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { DocumentDetailResponse, DocumentSummary, ExtractedEntity } from "@/types/api";
import {
  ENTITY_TYPE_STYLES,
  getAssertionStyle,
  getEntityText,
  getEntityTypeStyle,
} from "@/utils/entityStyles";

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
    const typeStyle = getEntityTypeStyle(ent.entity_type);
    const assertStyle = getAssertionStyle(ent.assertion_status, ent.temporal_status);
    nodes.push(
      <button
        key={ent.entity_id}
        type="button"
        data-entity-type={ent.entity_type}
        data-assertion={assertStyle.canonical}
        onClick={() => onSelect(ent.entity_id)}
        className={`inline-flex items-center gap-1 rounded px-1 py-0.5 text-left font-medium transition-colors cursor-pointer ${
          isSelected
            ? "bg-sky-600 text-white ring-2 ring-sky-300"
            : `${typeStyle.highlightClass} ${assertStyle.borderStyleClass}`
        }`}
        title={`${typeStyle.label} (${ent.entity_type}) · ${assertStyle.ariaLabel}`}
      >
        <span className={assertStyle.textClass}>{text.slice(ent.char_start, ent.char_end)}</span>
        <span className="text-[9px] font-bold opacity-80">{typeStyle.shortCode}</span>
        {assertStyle.badge && (
          <span className="rounded bg-slate-900/15 dark:bg-white/20 px-1 text-[9px] font-bold">
            {assertStyle.badge}
          </span>
        )}
        {assertStyle.showClock && <Clock className="h-2.5 w-2.5 inline" aria-label="Historical" />}
      </button>
    );
    cursor = ent.char_end;
  }
  if (cursor < text.length) {
    nodes.push(<span key={`t-end`}>{text.slice(cursor)}</span>);
  }

  return <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-800 dark:text-slate-100">{nodes}</pre>;
}

export function WorkspacePage() {
  const { t } = useTranslation();
  const [searchParams] = useSearchParams();
  const token = useAuthStore((s) => s.token);
  const queryClient = useQueryClient();
  const [selectedDocId, setSelectedDocId] = useState<string | null>(() => searchParams.get("doc"));
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(() => searchParams.get("entity"));
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
          <div className="rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-2 text-[11px] space-y-1.5" aria-label="Entity colour legend">
            <div className="flex flex-wrap gap-1">
              {Object.values(ENTITY_TYPE_STYLES).slice(0, 10).map((st) => (
                <span key={st.type} className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-medium ${st.chipClass}`}>
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: st.hex }} />
                  {st.type}
                </span>
              ))}
            </div>
            <div className="flex flex-wrap gap-2 text-[10px] text-slate-600 dark:text-slate-300">
              <span className="border-b-2 border-solid border-slate-500">Present: solid</span>
              <span className="line-through border-b-2 border-solid border-rose-500">Absent: NOT</span>
              <span className="border-b-2 border-dashed border-amber-500">Possible: ?</span>
              <span className="italic border-b-2 border-dotted border-indigo-500">Hypothetical: italic</span>
              <span>Family: FAM</span>
              <span><Clock className="h-2.5 w-2.5 inline mr-0.5" />Historical</span>
            </div>
          </div>
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
            detail?.entities.map((ent) => {
              const tStyle = getEntityTypeStyle(ent.entity_type);
              const aStyle = getAssertionStyle(ent.assertion_status, ent.temporal_status);
              return (
              <div
                key={ent.entity_id}
                onClick={() => setSelectedEntityId(ent.entity_id)}
                className={`rounded-md border p-2.5 text-xs cursor-pointer ${
                  ent.entity_id === selectedEntityId
                    ? "border-sky-400 bg-sky-50 dark:bg-sky-950/40"
                    : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className={`font-semibold text-slate-900 dark:text-slate-100 ${aStyle.textClass}`}>
                    {getEntityText(ent)}
                  </span>
                  <div className="flex items-center gap-1">
                    <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${tStyle.chipClass}`}>
                      {ent.entity_type}
                    </span>
                    <Badge tone={aStyle.canonical === "absent" ? "rose" : aStyle.canonical === "possible" ? "amber" : "emerald"}>
                      {aStyle.badge ? `${aStyle.badge} · ` : ""}{ent.assertion_status}
                    </Badge>
                  </div>
                </div>
                <div className="text-[11px] text-slate-600 dark:text-slate-300 mt-1">
                  {ent.entity_type} · {ent.normalized_text} · span [{ent.char_start}, {ent.char_end}]
                </div>
                <div className="flex items-center justify-between mt-2">
                  <div className="flex items-center gap-1">
                    <Badge tone={ent.review_status === "needs_review" ? "amber" : "slate"}>
                      {ent.review_status === "needs_review" ? "needs review" : ent.review_status}
                    </Badge>
                    {ent.review_reason ? (
                      <span className="text-[10px] text-amber-700 dark:text-amber-300">{ent.review_reason}</span>
                    ) : null}
                  </div>
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
              );
            })}

          {rightTab === "graph" && detail && (
            <div className="space-y-2">
              {detail.relations.length === 0 ? (
                <p className="text-xs text-slate-500">No relations extracted on this document.</p>
              ) : (
                detail.relations.map((rel) => {
                  const src = detail.entities.find((e) => e.entity_id === rel.source_entity_id);
                  const tgt = detail.entities.find((e) => e.entity_id === rel.target_entity_id);
                  return (
                    <div key={rel.relation_id} className="rounded border border-slate-200 dark:border-slate-700 p-2 text-xs">
                      <span className="font-medium">{getEntityText(src) || rel.source_entity_id}</span>{" "}
                      <Badge tone="sky">{rel.relation_type}</Badge>{" "}
                      <span className="font-medium">{getEntityText(tgt) || rel.target_entity_id}</span>
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
