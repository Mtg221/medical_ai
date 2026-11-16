import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, CheckCircle2, Eye, Flame, Link2, ShieldAlert, XCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest, fetchAuthenticatedBlobUrl } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { ImageFindingRecord, PatientUnifiedRecord } from "@/types/api";

const DX_CLASS_OPTIONS = [
  "melanoma",
  "basal_cell_carcinoma",
  "squamous_cell_carcinoma",
  "actinic_keratosis",
  "nevus",
  "seborrheic_keratosis",
  "dermatofibroma",
  "vascular_lesion",
  "other",
];

export function SkinVisionPage() {
  const { t, i18n } = useTranslation();
  const token = useAuthStore((s) => s.token);
  const queryClient = useQueryClient();

  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.55);
  const [correctedDx, setCorrectedDx] = useState<string>("melanoma");
  const [thumbUrl, setThumbUrl] = useState<string | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [patientId, setPatientId] = useState<string>("subj_synth_001");

  const imagesQuery = useQuery({
    queryKey: ["images"],
    queryFn: () => apiRequest<{ images: ImageFindingRecord[]; count: number }>("/api/v1/images", { token }),
    enabled: Boolean(token),
  });

  const activeImageId = selectedImageId ?? imagesQuery.data?.images[0]?.image_id ?? null;

  const detailQuery = useQuery({
    queryKey: ["image-detail", activeImageId],
    queryFn: () => apiRequest<ImageFindingRecord>(`/api/v1/images/${activeImageId}`, { token }),
    enabled: Boolean(token && activeImageId),
  });

  const patientRecordQuery = useQuery({
    queryKey: ["patient-record", patientId],
    queryFn: () => apiRequest<PatientUnifiedRecord>(`/api/v1/patients/${patientId}/record`, { token }),
    enabled: Boolean(token && patientId),
  });

  const finding = detailQuery.data ?? imagesQuery.data?.images.find((im) => im.image_id === activeImageId) ?? null;

  // Load authenticated thumbnail & Grad-CAM overlay images as Object URLs
  useEffect(() => {
    let cancelled = false;
    let tUrl: string | null = null;
    let hUrl: string | null = null;
    async function loadAssets() {
      if (!token || !activeImageId) {
        setThumbUrl(null);
        setHeatmapUrl(null);
        return;
      }
      try {
        tUrl = await fetchAuthenticatedBlobUrl(`/api/v1/images/${activeImageId}/thumbnail`, token);
        hUrl = await fetchAuthenticatedBlobUrl(`/api/v1/images/${activeImageId}/heatmap`, token);
        if (!cancelled) {
          setThumbUrl(tUrl);
          setHeatmapUrl(hUrl);
        }
      } catch {
        if (!cancelled) {
          setThumbUrl(null);
          setHeatmapUrl(null);
        }
      }
    }
    void loadAssets();
    return () => {
      cancelled = true;
      if (tUrl) URL.revokeObjectURL(tUrl);
      if (hUrl) URL.revokeObjectURL(hUrl);
    };
  }, [activeImageId, token]);

  const reviewMutation = useMutation({
    mutationFn: (payload: {
      review_decision: string;
      corrected_diagnosis?: string;
      rationale?: string;
    }) =>
      apiRequest<Record<string, unknown>>(`/api/v1/images/${activeImageId}/review`, {
        method: "POST",
        token,
        body: payload,
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["images"] });
      void queryClient.invalidateQueries({ queryKey: ["image-detail", activeImageId] });
      void queryClient.invalidateQueries({ queryKey: ["patient-record", patientId] });
    },
  });

  // Keyboard shortcuts: A (approve), R (reject), C (correct diagnosis)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (!activeImageId || reviewMutation.isPending) return;
      const k = e.key.toLowerCase();
      if (k === "a") {
        reviewMutation.mutate({ review_decision: "APPROVE_AS_GROUND_TRUTH" });
      } else if (k === "r") {
        reviewMutation.mutate({ review_decision: "REJECT" });
      } else if (k === "c") {
        reviewMutation.mutate({
          review_decision: "CORRECT_AND_APPROVE",
          corrected_diagnosis: correctedDx,
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeImageId, correctedDx, reviewMutation]);

  const isFr = i18n.language === "fr";
  const quality = finding?.quality_result;
  const riskBand = finding?.malignancy_risk_band ?? "abstained";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold text-slate-900 dark:text-slate-100">{t("skin.title")}</h1>
          <p className="text-xs text-amber-800 dark:text-amber-300 font-medium flex items-center gap-1.5 mt-1">
            <ShieldAlert className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>{t("skin.safetyNotice")}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <label htmlFor="patient-select" className="text-xs font-medium text-slate-600 dark:text-slate-300">
            Patient ID:
          </label>
          <input
            id="patient-select"
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="rounded border border-slate-300 dark:border-slate-700 dark:bg-slate-900 px-2.5 py-1 text-xs font-mono"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Thumbnail list */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-600 dark:text-slate-300">
              Skin Images ({imagesQuery.data?.count ?? 0})
            </h2>
          </CardHeader>
          <CardContent className="space-y-2 max-h-[65vh] overflow-y-auto">
            {(!imagesQuery.data || imagesQuery.data.images.length === 0) && (
              <p className="text-xs text-slate-500">{t("skin.noImages")}</p>
            )}
            {imagesQuery.data?.images.map((im) => {
              const active = im.image_id === activeImageId;
              return (
                <button
                  key={im.image_id}
                  type="button"
                  onClick={() => setSelectedImageId(im.image_id)}
                  className={`w-full text-left rounded-md border p-2.5 text-xs transition-colors ${
                    active
                      ? "border-sky-600 bg-sky-50 dark:bg-sky-950/50"
                      : "border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50"
                  }`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono font-semibold truncate">{im.image_id}</span>
                    <Badge
                      tone={
                        im.malignancy_risk_band === "high"
                          ? "rose"
                          : im.malignancy_risk_band === "medium"
                          ? "amber"
                          : im.malignancy_risk_band === "low"
                          ? "emerald"
                          : "slate"
                      }
                    >
                      {im.malignancy_risk_band}
                    </Badge>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                    <span>{im.picture_type}</span>
                    <span>{im.review_status}</span>
                  </div>
                </button>
              );
            })}
          </CardContent>
        </Card>

        {/* Center/Right: Interactive Image + Heatmap Viewer & Clinical Card */}
        <div className="lg:col-span-9 space-y-6">
          {finding ? (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Image + Grad-CAM Overlay Card */}
              <Card>
                <CardHeader className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Eye className="h-4 w-4 text-sky-700 dark:text-sky-400" aria-hidden="true" />
                    <span className="text-xs font-semibold">{finding.picture_type}</span>
                    {finding.body_site && <Badge tone="slate">{finding.body_site}</Badge>}
                  </div>
                  <label className="inline-flex items-center gap-1.5 text-xs font-medium cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showHeatmap}
                      onChange={(e) => setShowHeatmap(e.target.checked)}
                      className="rounded border-slate-300"
                    />
                    <Flame className="h-3.5 w-3.5 text-amber-600" aria-hidden="true" />
                    {t("skin.heatmapToggle")}
                  </label>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="relative w-full aspect-square max-h-80 rounded-lg overflow-hidden bg-slate-900 flex items-center justify-center border border-slate-200 dark:border-slate-800">
                    {thumbUrl ? (
                      <img
                        src={thumbUrl}
                        alt={`Skin lesion ${finding.image_id} (${finding.picture_type})`}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <span className="text-xs text-slate-400">Loading image...</span>
                    )}
                    {showHeatmap && heatmapUrl && (
                      <img
                        src={heatmapUrl}
                        alt={`Grad-CAM attention heatmap for ${finding.image_id}`}
                        data-testid="gradcam-heatmap-img"
                        style={{ opacity: heatmapOpacity }}
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                      />
                    )}
                  </div>

                  <div className="flex items-center gap-3 text-xs">
                    <label htmlFor="heatmap-opacity" className="font-medium text-slate-700 dark:text-slate-300 shrink-0">
                      {t("skin.heatmapOpacity")}: {Math.round(heatmapOpacity * 100)}%
                    </label>
                    <input
                      id="heatmap-opacity"
                      type="range"
                      min={0}
                      max={100}
                      value={Math.round(heatmapOpacity * 100)}
                      onChange={(e) => setHeatmapOpacity(Number(e.target.value) / 100)}
                      className="w-full"
                    />
                  </div>

                  {/* Quality Badges */}
                  {quality && (
                    <div className="rounded-md bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 p-3 space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{t("skin.qualityTitle")}</span>
                        <Badge tone={quality.quality_ok ? "emerald" : "rose"}>
                          {quality.quality_ok ? t("skin.qualityPass") : t("skin.qualityFail")} ({quality.quality_score.toFixed(2)})
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5 text-[11px] text-slate-600 dark:text-slate-300">
                        <span>Resolution: {quality.width}×{quality.height}px</span>
                        <span>Focus (Laplacian): {quality.laplacian_variance.toFixed(1)}</span>
                        <span>Brightness: {quality.mean_brightness.toFixed(2)}</span>
                        <span>Skin area: {Math.round(quality.skin_area_ratio * 100)}%</span>
                      </div>
                      {!quality.quality_ok && (
                        <p className="text-rose-700 dark:text-rose-300 text-[11px]">
                          {(isFr ? quality.reasons_fr : quality.reasons_en).join(" ")}
                        </p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Risk Estimate, Top-3 Diagnoses & Review Controls */}
              <Card>
                <CardHeader className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold">{t("skin.riskBand")}</span>
                  <div className="flex items-center gap-1.5">
                    {finding.review_status === "needs_review" && (
                      <Badge tone="amber">
                        <AlertCircle className="h-3 w-3 mr-1 inline" aria-hidden="true" />
                        {t("skin.needsReview")}
                      </Badge>
                    )}
                    <Badge tone={finding.review_status === "approved" || finding.review_status === "corrected" ? "emerald" : "slate"}>
                      {finding.review_status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4 text-xs">
                  <div
                    data-testid="skin-result-card"
                    className="rounded-md border border-slate-200 dark:border-slate-800 p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-slate-600 dark:text-slate-300">Malignancy Risk Band:</span>
                      <Badge
                        tone={
                          riskBand === "high"
                            ? "rose"
                            : riskBand === "medium"
                            ? "amber"
                            : riskBand === "low"
                            ? "emerald"
                            : "slate"
                        }
                      >
                        {riskBand.toUpperCase()}
                        {finding.malignancy_risk_probability !== null &&
                          ` (${(finding.malignancy_risk_probability * 100).toFixed(1)}%)`}
                      </Badge>
                    </div>
                    {finding.abstained && (
                      <p className="text-rose-700 dark:text-rose-300">
                        {isFr ? finding.abstain_reason_fr : finding.abstain_reason_en}
                      </p>
                    )}
                  </div>

                  {/* Top-3 Likely Diagnoses */}
                  <div className="space-y-1.5">
                    <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t("skin.top3Title")}</h3>
                    {finding.top3_diagnoses.length === 0 ? (
                      <p className="text-slate-500">No diagnosis candidates (abstained).</p>
                    ) : (
                      <ul className="space-y-1.5">
                        {finding.top3_diagnoses.map((cand) => (
                          <li key={cand.rank} className="space-y-0.5">
                            <div className="flex justify-between text-[11px]">
                              <span className="font-medium">
                                #{cand.rank} {cand.diagnosis}
                              </span>
                              <span className="font-mono">{(cand.probability * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full h-1.5 rounded bg-slate-100 dark:bg-slate-800 overflow-hidden">
                              <div
                                className="h-full bg-sky-600"
                                style={{ width: `${Math.min(100, cand.probability * 100)}%` }}
                              />
                            </div>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  {/* Review Controls */}
                  <div className="rounded-md bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 p-3 space-y-2.5">
                    <p className="font-semibold text-slate-800 dark:text-slate-200">{t("skin.reviewTitle")}</p>
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        size="sm"
                        onClick={() => reviewMutation.mutate({ review_decision: "APPROVE_AS_GROUND_TRUTH" })}
                        disabled={reviewMutation.isPending}
                      >
                        <CheckCircle2 className="h-3.5 w-3.5 mr-1" aria-hidden="true" />
                        {t("skin.approve")}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => reviewMutation.mutate({ review_decision: "REJECT" })}
                        disabled={reviewMutation.isPending}
                      >
                        <XCircle className="h-3.5 w-3.5 mr-1" aria-hidden="true" />
                        {t("skin.reject")}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => reviewMutation.mutate({ review_decision: "MARK_IMAGE_UNUSABLE" })}
                        disabled={reviewMutation.isPending}
                      >
                        {t("skin.markUnusable")}
                      </Button>
                    </div>
                    <div className="flex items-center gap-2">
                      <select
                        aria-label="Corrected diagnosis class"
                        value={correctedDx}
                        onChange={(e) => setCorrectedDx(e.target.value)}
                        className="rounded border border-slate-300 dark:border-slate-700 dark:bg-slate-900 px-2 py-1 text-xs flex-1"
                      >
                        {DX_CLASS_OPTIONS.map((dx) => (
                          <option key={dx} value={dx}>
                            {dx}
                          </option>
                        ))}
                      </select>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() =>
                          reviewMutation.mutate({
                            review_decision: "CORRECT_AND_APPROVE",
                            corrected_diagnosis: correctedDx,
                          })
                        }
                        disabled={reviewMutation.isPending}
                      >
                        {t("skin.correct")}
                      </Button>
                    </div>
                  </div>

                  {/* Linked Text Facts (SAME_LESION) */}
                  <div className="space-y-1.5">
                    <h3 className="font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                      <Link2 className="h-3.5 w-3.5 text-sky-700" aria-hidden="true" />
                      {t("skin.linkedTextTitle")}
                    </h3>
                    {(!finding.same_lesion_links || finding.same_lesion_links.length === 0) ? (
                      <p className="text-slate-500 text-[11px]">
                        No matching body-site text mentions linked yet.
                      </p>
                    ) : (
                      <ul className="space-y-1.5">
                        {finding.same_lesion_links.map((lnk) => (
                          <li
                            key={lnk.link_id}
                            className="rounded border border-sky-200 dark:border-sky-800 bg-sky-50/60 dark:bg-sky-950/30 p-2 text-[11px]"
                          >
                            <div className="flex justify-between font-medium">
                              <span>"{lnk.text_span}"</span>
                              <Badge tone="sky">conf {lnk.confidence.toFixed(2)}</Badge>
                            </div>
                            <p className="text-slate-600 dark:text-slate-400 mt-0.5">
                              {isFr ? lnk.rationale_fr : lnk.rationale_en}
                            </p>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : null}

          {/* Unified Patient Record & Timeline */}
          <Card>
            <CardHeader>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                {t("skin.patientRecordTitle")} ({patientRecordQuery.data?.pseudonym_code ?? patientId})
              </h2>
            </CardHeader>
            <CardContent className="space-y-2 text-xs max-h-64 overflow-y-auto">
              {patientRecordQuery.data?.timeline_events.length === 0 && (
                <p className="text-slate-500">No timeline events yet for this patient.</p>
              )}
              {patientRecordQuery.data?.timeline_events.map((evt) => (
                <div
                  key={evt.event_id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded border border-slate-200 dark:border-slate-800 px-3 py-2"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] text-slate-500">{evt.event_date_iso}</span>
                    <Badge tone={evt.event_category === "SKIN_IMAGE_FINDING" ? "sky" : "slate"}>
                      {evt.event_category}
                    </Badge>
                    <span>{evt.summary_text}</span>
                  </div>
                  {evt.review_status && <Badge tone="slate">{evt.review_status}</Badge>}
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
