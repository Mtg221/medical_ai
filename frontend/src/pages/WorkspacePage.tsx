import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Clock, Copy, FileSearch, Network, Code2, X } from "lucide-react";
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
import {
  buildClinicalRecordJson,
  buildSimpleKeyValueLines,
  isEntityApproved,
  isEntityRejected,
} from "@/utils/simpleOutput";

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
  const [simpleFormat, setSimpleFormat] = useState<"kv" | "json">("json");
  const [simpleApprovedOnly, setSimpleApprovedOnly] = useState<boolean>(false);
  const [copiedSimple, setCopiedSimple] = useState<boolean>(false);

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
  const simpleLines = buildSimpleKeyValueLines(detail, simpleApprovedOnly);
  const clinicalJson = buildClinicalRecordJson(detail, simpleApprovedOnly);
  const simpleOutputText =
    simpleFormat === "json"
      ? clinicalJson
        ? JSON.stringify(clinicalJson, null, 2)
        : ""
      : simpleLines.map((l) => `${l.key} = ${l.value}`).join("\n");

  const handleCopySimple = async () => {
    if (!simpleOutputText) return;
    try {
      await navigator.clipboard.writeText(simpleOutputText);
      setCopiedSimple(true);
      setTimeout(() => setCopiedSimple(false), 1800);
    } catch {
      // ignore clipboard denial
    }
  };

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
          {reviewMutation.isError && (
            <div role="alert" className="rounded border border-rose-300 bg-rose-50 px-2.5 py-1.5 text-xs text-rose-800 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-200">
              Review failed: {reviewMutation.error instanceof Error ? reviewMutation.error.message : "Unknown error"}
            </div>
          )}
          {rightTab === "entities" && detail && (
            <div className="rounded-lg border-2 border-emerald-200 bg-emerald-50/40 p-2.5 space-y-2 dark:border-emerald-900 dark:bg-emerald-950/20">
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] font-bold uppercase tracking-wide text-emerald-900 dark:text-emerald-200">
                    Structured Output
                  </span>
                  <div className="inline-flex rounded border border-emerald-300 dark:border-emerald-800 bg-white dark:bg-slate-900 p-0.5 text-[10px]">
                    <button
                      type="button"
                      onClick={() => setSimpleFormat("json")}
                      className={`rounded px-1.5 py-0.5 font-semibold ${
                        simpleFormat === "json"
                          ? "bg-emerald-600 text-white"
                          : "text-slate-600 dark:text-slate-300"
                      }`}
                    >
                      Clinical JSON
                    </button>
                    <button
                      type="button"
                      onClick={() => setSimpleFormat("kv")}
                      className={`rounded px-1.5 py-0.5 font-semibold ${
                        simpleFormat === "kv"
                          ? "bg-emerald-600 text-white"
                          : "text-slate-600 dark:text-slate-300"
                      }`}
                    >
                      key = value
                    </button>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <label className="inline-flex items-center gap-1 text-slate-700 dark:text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={simpleApprovedOnly}
                      onChange={(e) => setSimpleApprovedOnly(e.target.checked)}
                    />
                    Approved only
                  </label>
                  <button
                    type="button"
                    onClick={() => void handleCopySimple()}
                    disabled={!simpleOutputText}
                    className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-0.5 text-[11px] font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 disabled:opacity-50"
                  >
                    <Copy className="h-3 w-3" />
                    {copiedSimple ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>
              {simpleLines.length === 0 ? (
                <p className="text-[11px] text-slate-500">
                  {simpleApprovedOnly
                    ? "Click Approve on entities below to populate the verified output."
                    : "No active entities."}
                </p>
              ) : (
                <pre
                  data-testid="workspace-simple-kv-output"
                  className="rounded border border-emerald-200/80 bg-white dark:border-emerald-900/60 dark:bg-slate-900 px-2.5 py-2 font-mono text-xs leading-5 text-slate-900 dark:text-slate-100 whitespace-pre-wrap select-all max-h-44 overflow-y-auto"
                >
                  {simpleOutputText}
                </pre>
              )}
            </div>
          )}

          {rightTab === "entities" &&
            detail?.entities.map((ent) => {
              const tStyle = getEntityTypeStyle(ent.entity_type);
              const aStyle = getAssertionStyle(ent.assertion_status, ent.temporal_status);
              const approved = isEntityApproved(ent.review_status);
              const rejected = isEntityRejected(ent.review_status);
              const statusBadgeLabel = approved
                ? "✓ Approved"
                : rejected
                ? "✗ Rejected"
                : "Pending Review";
              const statusBadgeTone = approved ? "emerald" : rejected ? "rose" : "amber";
              return (
              <div
                key={ent.entity_id}
                onClick={() => setSelectedEntityId(ent.entity_id)}
                className={`rounded-md border p-2.5 text-xs cursor-pointer transition-colors ${
                  ent.entity_id === selectedEntityId
                    ? "border-sky-400 bg-sky-50 dark:bg-sky-950/40"
                    : approved
                    ? "border-emerald-300 bg-emerald-50/50 dark:border-emerald-800 dark:bg-emerald-950/25"
                    : rejected
                    ? "border-rose-300 bg-rose-50/50 dark:border-rose-800 dark:bg-rose-950/25"
                    : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`font-semibold ${
                      rejected
                        ? "line-through text-rose-800 dark:text-rose-300"
                        : `text-slate-900 dark:text-slate-100 ${aStyle.textClass}`
                    }`}
                  >
                    {getEntityText(ent)}
                  </span>
                  <div className="flex items-center gap-1">
                    <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${tStyle.chipClass}`}>
                      {ent.entity_type}
                    </span>
                    <Badge tone={statusBadgeTone}>{statusBadgeLabel}</Badge>
                  </div>
                </div>
                <div className="text-[11px] text-slate-600 dark:text-slate-300 mt-1">
                  {ent.certainty === "absent" || ent.assertion_status === "NEGATED"
                    ? "Absent / Negated"
                    : ent.certainty === "possible" || ent.assertion_status === "UNCERTAIN"
                    ? "Possible / Uncertain"
                    : "Confirmed"}{" "}
                  · {ent.temporality === "historical" ? "Historical" : "Current"} ·{" "}
                  {ent.experiencer === "family" ? "Family" : "Patient"}
                  {ent.review_reason ? ` · ${ent.review_reason}` : ""}
                </div>
                <div className="flex items-center justify-end gap-1.5 mt-2">
                  <button
                    type="button"
                    disabled={reviewMutation.isPending}
                    onClick={(e) => {
                      e.stopPropagation();
                      reviewMutation.mutate({ entityId: ent.entity_id, decision: "APPROVE_AS_GROUND_TRUTH" });
                    }}
                    className={`inline-flex items-center gap-1 rounded px-2.5 py-1 text-xs font-semibold transition-colors ${
                      approved
                        ? "bg-emerald-700 text-white ring-2 ring-emerald-400"
                        : "bg-emerald-600 text-white hover:bg-emerald-700"
                    } disabled:opacity-50`}
                  >
                    <Check className="h-3 w-3" aria-hidden="true" />
                    {t("workspace.approve")}
                  </button>
                  <button
                    type="button"
                    disabled={reviewMutation.isPending}
                    onClick={(e) => {
                      e.stopPropagation();
                      reviewMutation.mutate({ entityId: ent.entity_id, decision: "REJECT" });
                    }}
                    className={`inline-flex items-center gap-1 rounded px-2.5 py-1 text-xs font-semibold transition-colors ${
                      rejected
                        ? "bg-rose-700 text-white ring-2 ring-rose-400"
                        : "bg-rose-600 text-white hover:bg-rose-700"
                    } disabled:opacity-50`}
                  >
                    <X className="h-3 w-3" aria-hidden="true" />
                    {t("workspace.reject")}
                  </button>
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
