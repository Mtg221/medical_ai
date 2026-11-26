import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Download, Edit3, Flag, X } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import { getEntityText } from "@/utils/entityStyles";
import type {
  DocumentDetailResponse,
  DocumentSummary,
  ExtractedEntity,
  HumanReviewRecord,
  ReviewSubmissionRequest,
} from "@/types/api";

const ENTITY_TYPES = [
  "DIAGNOSIS",
  "SYMPTOM",
  "MEDICATION",
  "DOSAGE",
  "FREQUENCY",
  "ROUTE",
  "LAB_TEST",
  "LAB_VALUE",
  "LAB_UNIT",
  "PROCEDURE",
  "ANATOMY",
  "TEMPORAL_EXPR",
] as const;

const ASSERTION_OPTIONS = [
  "AFFIRMED",
  "NEGATED",
  "UNCERTAIN",
  "HYPOTHETICAL",
  "CONDITIONAL",
  "FAMILY_HISTORY",
] as const;

const CERTAINTY_OPTIONS = ["confirmed", "possible", "absent"] as const;
const TEMPORALITY_OPTIONS = ["current", "historical"] as const;
const EXPERIENCER_OPTIONS = ["patient", "family"] as const;

export function ReviewQueuePage() {
  const token = useAuthStore((s) => s.token);
  const queryClient = useQueryClient();
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [onlyFlagged, setOnlyFlagged] = useState<boolean>(false);
  const [focusedIdx, setFocusedIdx] = useState<number>(0);
  const [editingEntity, setEditingEntity] = useState<ExtractedEntity | null>(null);
  const [corrType, setCorrType] = useState<string>("DIAGNOSIS");
  const [corrAssertion, setCorrAssertion] = useState<string>("AFFIRMED");
  const [corrCertainty, setCorrCertainty] = useState<"confirmed" | "possible" | "absent">("confirmed");
  const [corrTemporality, setCorrTemporality] = useState<"current" | "historical">("current");
  const [corrExperiencer, setCorrExperiencer] = useState<"patient" | "family">("patient");
  const [corrStart, setCorrStart] = useState<number>(0);
  const [corrEnd, setCorrEnd] = useState<number>(0);
  const [corrSurface, setCorrSurface] = useState<string>("");
  const [corrRationale, setCorrRationale] = useState<string>("Expert clinical review");
  const [statusMessage, setStatusMessage] = useState<string>("");

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

  const reviewsQuery = useQuery({
    queryKey: ["reviews", activeDocId],
    queryFn: () =>
      apiRequest<{ reviews: HumanReviewRecord[]; count: number }>(
        `/api/v1/reviews?document_id=${encodeURIComponent(activeDocId)}`,
        { token }
      ),
    enabled: Boolean(token && activeDocId),
  });

  const reviewMutation = useMutation({
    mutationFn: (payload: ReviewSubmissionRequest) =>
      apiRequest<{ review_id: string; review_status: string }>("/api/v1/reviews", {
        method: "POST",
        token,
        body: payload,
      }),
    onSuccess: (res) => {
      setStatusMessage(`Saved ${res.review_status} (${res.review_id})`);
      setEditingEntity(null);
      void queryClient.invalidateQueries({ queryKey: ["document-detail", activeDocId] });
      void queryClient.invalidateQueries({ queryKey: ["reviews", activeDocId] });
    },
    onError: (err: Error) => {
      setStatusMessage(`Error: ${err.message}`);
    },
  });

  const allEntities = detailQuery.data?.entities ?? [];
  const entities = allEntities
    .filter((e) =>
      onlyFlagged
        ? e.review_status === "needs_review" ||
          e.confidence < 0.75 ||
          e.review_status === "UNRESOLVED_DISAGREEMENT"
        : true
    )
    .slice()
    .sort((a, b) => {
      const aNeeds = a.review_status === "needs_review" ? 0 : 1;
      const bNeeds = b.review_status === "needs_review" ? 0 : 1;
      if (aNeeds !== bNeeds) return aNeeds - bNeeds;
      return a.confidence - b.confidence;
    });
  const reviewedCount = allEntities.filter(
    (e) => e.review_status !== "AI_PREDICTED" && e.review_status !== "needs_review"
  ).length;
  const progressPct = allEntities.length > 0 ? Math.round((reviewedCount / allEntities.length) * 100) : 0;

  const docText =
    detailQuery.data?.document.deidentified_preview ||
    detailQuery.data?.pages.map((p) => p.raw_text).join("\n") ||
    "";

  const openCorrection = (ent: ExtractedEntity) => {
    setEditingEntity(ent);
    setCorrType(ent.entity_type);
    setCorrAssertion(ent.assertion_status);
    setCorrCertainty(ent.certainty ?? "confirmed");
    setCorrTemporality(ent.temporality ?? "current");
    setCorrExperiencer(ent.experiencer ?? "patient");
    setCorrStart(ent.char_start);
    setCorrEnd(ent.char_end);
    setCorrSurface(getEntityText(ent));
    setCorrRationale("Span / label correction");
  };

  const handleTextSelection = () => {
    if (!editingEntity || typeof window === "undefined") return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;
    const idx = docText.indexOf(selectedText);
    if (idx >= 0) {
      setCorrStart(idx);
      setCorrEnd(idx + selectedText.length);
      setCorrSurface(selectedText);
    }
  };

  const handleExport = async (format: "jsonl" | "brat") => {
    if (!token) return;
    const resp = await fetch(
      `/api/v1/reviews/export?format=${format}&document_id=${encodeURIComponent(activeDocId)}`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    if (!resp.ok) return;
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `reviewed_gold_${activeDocId}.${format === "brat" ? "ann" : "jsonl"}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    const onKeyDown = (ev: KeyboardEvent) => {
      const target = ev.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (entities.length === 0) return;
      const current = entities[Math.min(focusedIdx, entities.length - 1)];
      if (!current) return;
      const key = ev.key.toLowerCase();
      if (key === "a") {
        ev.preventDefault();
        reviewMutation.mutate({ entity_id: current.entity_id, action: "approve", rationale: "Shortcut A" });
      } else if (key === "r") {
        ev.preventDefault();
        reviewMutation.mutate({ entity_id: current.entity_id, action: "reject", rationale: "Shortcut R" });
      } else if (key === "c") {
        ev.preventDefault();
        openCorrection(current);
      } else if (key === "n") {
        ev.preventDefault();
        setFocusedIdx((idx) => (idx + 1) % entities.length);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [entities, focusedIdx, reviewMutation]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Human-in-the-Loop Review Mode
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Shortcuts: <kbd className="rounded border px-1">A</kbd> Approve ·{" "}
              <kbd className="rounded border px-1">C</kbd> Correct ·{" "}
              <kbd className="rounded border px-1">R</kbd> Reject ·{" "}
              <kbd className="rounded border px-1">N</kbd> Next
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="inline-flex items-center gap-1.5 text-xs text-slate-700 dark:text-slate-300">
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
              onChange={(e) => {
                setSelectedDocId(e.target.value);
                setFocusedIdx(0);
                setEditingEntity(null);
              }}
              className="rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            >
              {docsQuery.data?.documents.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  {d.original_filename || d.document_id}
                </option>
              ))}
            </select>
            <Button size="sm" variant="outline" onClick={() => void handleExport("jsonl")}>
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              Export JSONL
            </Button>
            <Button size="sm" variant="outline" onClick={() => void handleExport("brat")}>
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              Export BRAT
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress bar */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-300">
              <span>
                Review progress: {reviewedCount} / {allEntities.length} entities ({progressPct}%)
              </span>
              {statusMessage && <span className="font-medium text-sky-700 dark:text-sky-300">{statusMessage}</span>}
            </div>
            <div
              role="progressbar"
              aria-valuenow={progressPct}
              aria-valuemin={0}
              aria-valuemax={100}
              className="h-2 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden"
            >
              <div
                className="h-full bg-emerald-600 transition-all"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>

          {/* Document text panel (selectable for span correction) */}
          {docText && (
            <div className="rounded border border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed dark:border-slate-700 dark:bg-slate-800/60">
              <div className="mb-1 font-medium text-slate-700 dark:text-slate-200">
                Document Text (highlight substring while editing to update span offsets)
              </div>
              <pre
                onMouseUp={handleTextSelection}
                className="whitespace-pre-wrap font-sans text-slate-800 dark:text-slate-100 select-text"
              >
                {docText}
              </pre>
            </div>
          )}

          {/* Correction form */}
          {editingEntity && (
            <form
              className="rounded border border-sky-300 bg-sky-50/70 p-3 space-y-3 text-xs dark:border-sky-800 dark:bg-sky-950/40"
              onSubmit={(ev) => {
                ev.preventDefault();
                reviewMutation.mutate({
                  entity_id: editingEntity.entity_id,
                  action: "correct",
                  corrected_entity_type: corrType,
                  corrected_assertion_status: corrAssertion,
                  corrected_certainty: corrCertainty,
                  corrected_temporality: corrTemporality,
                  corrected_experiencer: corrExperiencer,
                  corrected_char_start: corrStart,
                  corrected_char_end: corrEnd,
                  corrected_surface_text: corrSurface,
                  rationale: corrRationale,
                });
              }}
            >
              <div className="font-semibold text-sky-900 dark:text-sky-200">
                Correct Entity: {getEntityText(editingEntity)}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <label className="flex flex-col gap-1">
                  <span>Entity Type</span>
                  <select
                    aria-label="Corrected entity type"
                    value={corrType}
                    onChange={(e) => setCorrType(e.target.value)}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  >
                    {ENTITY_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span>Assertion</span>
                  <select
                    aria-label="Corrected assertion status"
                    value={corrAssertion}
                    onChange={(e) => setCorrAssertion(e.target.value)}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  >
                    {ASSERTION_OPTIONS.map((a) => (
                      <option key={a} value={a}>
                        {a}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span>Certainty</span>
                  <select
                    aria-label="Corrected certainty"
                    value={corrCertainty}
                    onChange={(e) => setCorrCertainty(e.target.value as "confirmed" | "possible" | "absent")}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  >
                    {CERTAINTY_OPTIONS.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span>Temporality</span>
                  <select
                    aria-label="Corrected temporality"
                    value={corrTemporality}
                    onChange={(e) => setCorrTemporality(e.target.value as "current" | "historical")}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  >
                    {TEMPORALITY_OPTIONS.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span>Experiencer</span>
                  <select
                    aria-label="Corrected experiencer"
                    value={corrExperiencer}
                    onChange={(e) => setCorrExperiencer(e.target.value as "patient" | "family")}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  >
                    {EXPERIENCER_OPTIONS.map((x) => (
                      <option key={x} value={x}>
                        {x}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span>Surface Text</span>
                  <input
                    aria-label="Corrected surface text"
                    value={corrSurface}
                    onChange={(e) => setCorrSurface(e.target.value)}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span>Char Start</span>
                  <input
                    type="number"
                    aria-label="Corrected char start"
                    value={corrStart}
                    onChange={(e) => setCorrStart(Number(e.target.value))}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span>Char End</span>
                  <input
                    type="number"
                    aria-label="Corrected char end"
                    value={corrEnd}
                    onChange={(e) => setCorrEnd(Number(e.target.value))}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span>Rationale</span>
                  <input
                    aria-label="Correction rationale"
                    value={corrRationale}
                    onChange={(e) => setCorrRationale(e.target.value)}
                    className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800"
                  />
                </label>
              </div>
              <div className="flex items-center gap-2">
                <Button type="submit" size="sm" disabled={reviewMutation.isPending}>
                  Save Correction
                </Button>
                <Button type="button" size="sm" variant="outline" onClick={() => setEditingEntity(null)}>
                  Cancel
                </Button>
              </div>
            </form>
          )}

          {/* Clear, explicit 1-line decision legend */}
          <div className="rounded-lg border border-slate-200 bg-slate-50/80 px-3 py-2 text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-800/50 dark:text-slate-200 flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="font-semibold text-slate-900 dark:text-slate-100">Quick Guide:</span>
            <span>
              <strong className="text-emerald-700 dark:text-emerald-400">✓ Approve</strong> = extraction is correct
            </span>
            <span>
              <strong className="text-rose-700 dark:text-rose-400">✗ Reject</strong> = wrong / false positive
            </span>
            <span>
              <strong className="text-sky-700 dark:text-sky-400">✎ Edit</strong> = fix type, negation, or span
            </span>
            <span>
              <strong className="text-amber-700 dark:text-amber-400">⚑ Flag</strong> = second opinion needed
            </span>
          </div>

          {/* Entity list */}
          {entities.length === 0 ? (
            <p className="text-xs text-slate-500 py-6 text-center">No entities match the current filter.</p>
          ) : (
            <div className="space-y-2.5">
              {entities.map((ent, idx) => {
                const isFocused = idx === focusedIdx;
                const labelText = getEntityText(ent);
                const statusUpper = String(ent.review_status || "").toUpperCase();
                const isApproved = statusUpper === "APPROVED";
                const isRejected = statusUpper === "REJECTED";
                const isCorrected = statusUpper === "CORRECTED";
                const isFlagged = statusUpper === "FLAGGED";

                const cardTone = isApproved
                  ? "border-emerald-300 bg-emerald-50/50 dark:border-emerald-800 dark:bg-emerald-950/25"
                  : isRejected
                  ? "border-rose-300 bg-rose-50/50 dark:border-rose-800 dark:bg-rose-950/25"
                  : isCorrected
                  ? "border-sky-300 bg-sky-50/50 dark:border-sky-800 dark:bg-sky-950/25"
                  : isFocused
                  ? "border-sky-500 bg-sky-50/30 dark:border-sky-400 dark:bg-sky-950/30"
                  : "border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900/40";

                const statusLabel = isApproved
                  ? "✓ Approved"
                  : isRejected
                  ? "✗ Rejected"
                  : isCorrected
                  ? "✎ Corrected"
                  : isFlagged
                  ? "⚑ Flagged"
                  : "Pending Review";

                const statusTone = isApproved
                  ? "emerald"
                  : isRejected
                  ? "rose"
                  : isCorrected
                  ? "sky"
                  : "amber";

                const certaintyText =
                  ent.certainty === "absent" || ent.assertion_status === "NEGATED"
                    ? "Absent / Negated"
                    : ent.certainty === "possible" || ent.assertion_status === "UNCERTAIN"
                    ? "Possible / Uncertain"
                    : "Confirmed";

                const contextParts = [
                  certaintyText,
                  ent.temporality === "historical" ? "Historical" : "Current",
                  ent.experiencer === "family" ? "Family History" : "Patient",
                ];

                return (
                  <div
                    key={ent.entity_id}
                    onClick={() => setFocusedIdx(idx)}
                    className={`rounded-lg border p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs cursor-pointer transition-colors ${cardTone}`}
                  >
                    <div className="space-y-1.5 min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`text-sm font-semibold ${
                            isRejected
                              ? "line-through text-rose-800 dark:text-rose-300"
                              : "text-slate-900 dark:text-slate-100"
                          }`}
                        >
                          “{labelText}”
                        </span>
                        <Badge tone="sky">{ent.entity_type}</Badge>
                        <Badge tone={statusTone}>{statusLabel}</Badge>
                      </div>
                      <div className="text-slate-600 dark:text-slate-300 flex flex-wrap items-center gap-x-2 gap-y-1">
                        <span>{contextParts.join(" · ")}</span>
                        <span className="text-slate-400">•</span>
                        <span>Confidence: <strong>{(ent.confidence * 100).toFixed(0)}%</strong></span>
                        <span className="text-slate-400">•</span>
                        <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400">
                          chars {ent.char_start}–{ent.char_end}
                        </span>
                        {ent.review_reason ? (
                          <span className="text-amber-700 dark:text-amber-300 font-medium">
                            ({ent.review_reason})
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        disabled={reviewMutation.isPending}
                        onClick={(ev) => {
                          ev.stopPropagation();
                          reviewMutation.mutate({ entity_id: ent.entity_id, action: "approve", rationale: "Approved in UI" });
                        }}
                        aria-label={`Approve ${labelText}`}
                        className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
                          isApproved
                            ? "bg-emerald-700 text-white ring-2 ring-emerald-400"
                            : "bg-emerald-600 text-white hover:bg-emerald-700"
                        } disabled:opacity-50`}
                      >
                        <Check className="h-3.5 w-3.5" />
                        Approve
                      </button>
                      <button
                        type="button"
                        disabled={reviewMutation.isPending}
                        onClick={(ev) => {
                          ev.stopPropagation();
                          reviewMutation.mutate({ entity_id: ent.entity_id, action: "reject", rationale: "Rejected in UI" });
                        }}
                        aria-label={`Reject ${labelText}`}
                        className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
                          isRejected
                            ? "bg-rose-700 text-white ring-2 ring-rose-400"
                            : "bg-rose-600 text-white hover:bg-rose-700"
                        } disabled:opacity-50`}
                      >
                        <X className="h-3.5 w-3.5" />
                        Reject
                      </button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(ev) => {
                          ev.stopPropagation();
                          openCorrection(ent);
                        }}
                        aria-label={`Correct ${labelText}`}
                      >
                        <Edit3 className="h-3.5 w-3.5 text-sky-700" />
                        Edit
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(ev) => {
                          ev.stopPropagation();
                          reviewMutation.mutate({ entity_id: ent.entity_id, action: "flag", rationale: "Flagged in UI" });
                        }}
                        aria-label={`Flag ${labelText}`}
                      >
                        <Flag className="h-3.5 w-3.5 text-amber-600" />
                        Flag
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Persisted audit log */}
          {(reviewsQuery.data?.reviews.length ?? 0) > 0 && (
            <div className="pt-2 border-t border-slate-200 dark:border-slate-700 space-y-1.5">
              <h2 className="text-xs font-semibold text-slate-700 dark:text-slate-200">
                Saved Review Audit Log ({reviewsQuery.data?.count})
              </h2>
              <div className="space-y-1 max-h-40 overflow-y-auto text-xs">
                {reviewsQuery.data?.reviews.map((rev) => (
                  <div
                    key={rev.review_id}
                    className="rounded border border-slate-200 px-2.5 py-1.5 flex flex-wrap justify-between gap-2 dark:border-slate-700"
                  >
                    <span>
                      <strong>{rev.review_decision}</strong> · {String(rev.previous_value.entity_type ?? "")} →{" "}
                      {String(rev.corrected_value.entity_type ?? "")} ({rev.rationale})
                    </span>
                    <span className="text-slate-500 dark:text-slate-400">
                      {rev.reviewer_user_id} · {rev.created_at.slice(0, 19)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
