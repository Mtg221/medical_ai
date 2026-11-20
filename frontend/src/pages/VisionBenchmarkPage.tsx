import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";

type SubgroupAxis = "by_skin_tone" | "by_modality" | "by_dataset" | "by_body_site";

interface SubgroupEntry {
  sample_count: number;
  malignant_count: number;
  benign_count: number;
  auroc: number;
  auroc_ci95: [number, number];
  sensitivity_at_90_specificity: number;
  balanced_accuracy: number;
  ece: number;
  small_sample_flag: string;
}

interface ModelMetricsBlock {
  sample_count: number;
  auroc: number;
  auroc_ci95: [number, number];
  sensitivity_at_90_specificity: number;
  sensitivity_at_95_specificity: number;
  balanced_accuracy: number;
  diagnosis_macro_f1: number;
  diagnosis_top3_accuracy: number;
  ece: number;
  brier_score: number;
  abstention_rate: number;
  non_abstained_high_risk_precision: number;
  roc_curve?: Array<{ fpr: number; tpr: number; threshold: number }>;
  reliability_bins?: Array<{ bin: string; mean_predicted_prob: number; empirical_positive_rate: number; count: number }>;
  subgroups?: Record<SubgroupAxis, Record<string, SubgroupEntry>>;
}

interface ModelEvalBlock extends Partial<ModelMetricsBlock> {
  training?: Record<string, unknown>;
  locked_test?: ModelMetricsBlock;
}

interface RouterEvalBlock {
  accuracy: number;
  macro_f1?: number;
  document_photos_sent_to_skin_model?: number;
  classes?: string[];
  critical_errors?: {
    document_scan_to_skin?: number;
  };
  confusion_matrix?:
    | number[][]
    | {
        labels: string[];
        matrix: number[][];
      };
}

interface VisionSummaryPayload {
  router_evaluation?: RouterEvalBlock & {
    locked_test_evaluation?: RouterEvalBlock;
  };
  skin_benchmark?: {
    dataset_splits?: Record<string, number>;
    models?: Record<string, ModelEvalBlock>;
    seed_runs?: Record<string, { elapsed_seconds?: number }>;
    no_meta_train_info?: { elapsed_seconds?: number };
    ablations?: Array<Record<string, unknown>>;
    paired_bootstrap_holm?: Array<{
      reference_model: string;
      comparator_model: string;
      reference_auroc: number;
      comparator_auroc: number;
      delta_auroc: number;
      p_value_holm: number;
      significant_at_0_05: boolean;
    }>;
  };
}

function resolveModelMetrics(block?: ModelEvalBlock): ModelMetricsBlock | undefined {
  if (!block) return undefined;
  if (block.locked_test) return block.locked_test;
  if (typeof block.auroc === "number") return block as ModelMetricsBlock;
  return undefined;
}

export function VisionBenchmarkPage() {
  const { t } = useTranslation();
  const token = useAuthStore((s) => s.token);
  const [subgroupAxis, setSubgroupAxis] = useState<SubgroupAxis>("by_skin_tone");

  const benchQuery = useQuery({
    queryKey: ["vision-results"],
    queryFn: () => apiRequest<VisionSummaryPayload>("/api/v1/results/vision/summary", { token }),
    enabled: Boolean(token),
  });

  const skinModels = benchQuery.data?.skin_benchmark?.models ?? {};
  const primaryModelBlock =
    skinModels.pytorch_se_resnet_224_3seed_tta_ensemble ??
    skinModels.pytorch_resnet_3seed_ensemble ??
    skinModels.pytorch_resnet_seed1 ??
    skinModels.derm_cnn_multitask ??
    Object.values(skinModels)[0];
  const cnnTest = resolveModelMetrics(primaryModelBlock);
  const rocData = cnnTest?.roc_curve ?? [];
  const calData = (cnnTest?.reliability_bins ?? []).filter((b) => b.count > 0);
  const subgroupMap = cnnTest?.subgroups?.[subgroupAxis] ?? {};
  const subgroupRows = Object.entries(subgroupMap).map(([name, stats]) => ({
    name,
    ...stats,
  }));

  const routerRaw = benchQuery.data?.router_evaluation;
  const routerTest = routerRaw?.locked_test_evaluation ?? routerRaw;
  const routerCmLabels = Array.isArray(routerTest?.confusion_matrix)
    ? routerTest?.classes ?? []
    : routerTest?.confusion_matrix?.labels ?? [];
  const routerCmMatrix = Array.isArray(routerTest?.confusion_matrix)
    ? routerTest.confusion_matrix
    : routerTest?.confusion_matrix?.matrix ?? [];
  const routerDocsToSkin =
    routerTest?.document_photos_sent_to_skin_model ?? routerTest?.critical_errors?.document_scan_to_skin ?? 0;

  const ablations = benchQuery.data?.skin_benchmark?.ablations ?? [];
  const pairedBootstrap = benchQuery.data?.skin_benchmark?.paired_bootstrap_holm ?? [];
  const seedRuns = benchQuery.data?.skin_benchmark?.seed_runs ?? {};
  const noMetaInfo = benchQuery.data?.skin_benchmark?.no_meta_train_info;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-bold text-slate-900 dark:text-slate-100">{t("visionBench.title")}</h1>
        <p className="text-xs text-slate-600 dark:text-slate-400">{t("visionBench.subtitle")}</p>
      </div>

      {/* Model Comparison & Training Cost */}
      <Card>
        <CardHeader>
          <h2 className="text-sm font-semibold">From-Scratch Skin Models (Locked Test)</h2>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300">
                <th className="py-2 pr-3">Model</th>
                <th className="py-2 px-2">AUROC [95% CI]</th>
                <th className="py-2 px-2">Sens @ 90% Spec</th>
                <th className="py-2 px-2">Sens @ 95% Spec</th>
                <th className="py-2 px-2">Bal Acc</th>
                <th className="py-2 px-2">Dx Macro-F1</th>
                <th className="py-2 px-2">Top-3 Acc</th>
                <th className="py-2 px-2">ECE / Brier</th>
                <th className="py-2 px-2">Abstain</th>
                <th className="py-2 pl-2">Train Time</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(skinModels).map(([mName, block]) => {
                const lt = resolveModelMetrics(block);
                const tr = block.training;
                if (!lt) return null;
                let trainSec: unknown = tr?.training_seconds;
                if (trainSec == null && mName === "pytorch_resnet_seed1") {
                  trainSec = seedRuns.seed1?.elapsed_seconds;
                } else if (trainSec == null && mName === "pytorch_resnet_no_meta") {
                  trainSec = noMetaInfo?.elapsed_seconds;
                }
                return (
                  <tr key={mName} className="border-b border-slate-100 dark:border-slate-800/70">
                    <td className="py-2 pr-3 font-mono font-semibold">{mName}</td>
                    <td className="py-2 px-2">
                      {lt.auroc.toFixed(4)} [{lt.auroc_ci95[0].toFixed(3)}, {lt.auroc_ci95[1].toFixed(3)}]
                    </td>
                    <td className="py-2 px-2">{lt.sensitivity_at_90_specificity.toFixed(4)}</td>
                    <td className="py-2 px-2">{lt.sensitivity_at_95_specificity.toFixed(4)}</td>
                    <td className="py-2 px-2">{lt.balanced_accuracy.toFixed(4)}</td>
                    <td className="py-2 px-2">{lt.diagnosis_macro_f1.toFixed(4)}</td>
                    <td className="py-2 px-2">{lt.diagnosis_top3_accuracy.toFixed(4)}</td>
                    <td className="py-2 px-2">
                      {lt.ece.toFixed(4)} / {lt.brier_score.toFixed(4)}
                    </td>
                    <td className="py-2 px-2">{(lt.abstention_rate * 100).toFixed(1)}%</td>
                    <td className="py-2 pl-2 font-mono">{trainSec != null ? `${String(trainSec)}s` : "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* ROC Curve + Calibration Reliability Diagram */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <h2 className="text-sm font-semibold">{t("visionBench.rocTitle")}</h2>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "FPR", position: "insideBottom", offset: -4 }} />
                <YAxis type="number" domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line type="monotone" dataKey="tpr" name="DermResNetMultiTask TPR" stroke="#0284c7" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h2 className="text-sm font-semibold">{t("visionBench.calibrationTitle")}</h2>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={calData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="mean_predicted_prob" name="Predicted Prob" stroke="#64748b" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="empirical_positive_rate" name="Empirical Positive Rate" stroke="#0d9488" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Subgroup Fairness Charts & Table */}
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">{t("visionBench.subgroupTitle")}</h2>
          <div className="flex flex-wrap gap-1.5">
            {(["by_skin_tone", "by_modality", "by_dataset", "by_body_site"] as SubgroupAxis[]).map((ax) => (
              <button
                key={ax}
                type="button"
                onClick={() => setSubgroupAxis(ax)}
                className={`rounded px-2.5 py-1 text-xs font-medium border ${
                  subgroupAxis === ax
                    ? "bg-sky-700 text-white border-sky-700"
                    : "border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300"
                }`}
              >
                {ax.replace("by_", "")}
              </button>
            ))}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={subgroupRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="auroc" name="AUROC" fill="#0284c7" />
                <Bar dataKey="balanced_accuracy" name="Balanced Acc" fill="#0d9488" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800">
                  <th className="py-1.5 pr-3">Subgroup</th>
                  <th className="py-1.5 px-2">n (mal / ben)</th>
                  <th className="py-1.5 px-2">AUROC [95% CI]</th>
                  <th className="py-1.5 px-2">Sens @ 90% Spec</th>
                  <th className="py-1.5 px-2">ECE</th>
                  <th className="py-1.5 pl-2">Sample Size Flag</th>
                </tr>
              </thead>
              <tbody>
                {subgroupRows.map((row) => (
                  <tr key={row.name} className="border-b border-slate-100 dark:border-slate-800/70">
                    <td className="py-1.5 pr-3 font-mono font-medium">{row.name}</td>
                    <td className="py-1.5 px-2">
                      {row.sample_count} ({row.malignant_count} / {row.benign_count})
                    </td>
                    <td className="py-1.5 px-2">
                      {row.auroc.toFixed(4)} [{row.auroc_ci95[0].toFixed(3)}, {row.auroc_ci95[1].toFixed(3)}]
                    </td>
                    <td className="py-1.5 px-2">{row.sensitivity_at_90_specificity.toFixed(4)}</td>
                    <td className="py-1.5 px-2">{row.ece.toFixed(4)}</td>
                    <td className="py-1.5 pl-2">
                      <Badge tone={row.sample_count < 30 ? "amber" : "emerald"}>
                        {row.sample_count < 30 ? t("visionBench.tooFew") : row.small_sample_flag}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Router Confusion Matrix + Ablations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">{t("visionBench.routerTitle")}</h2>
            {routerTest && typeof routerTest.accuracy === "number" && (
              <Badge tone="emerald">
                Acc {(routerTest.accuracy * 100).toFixed(1)}% · Docs→Skin: {routerDocsToSkin}
              </Badge>
            )}
          </CardHeader>
          <CardContent className="overflow-x-auto">
            {routerCmLabels.length > 0 && routerCmMatrix.length > 0 && (
              <table className="w-full text-[11px] text-center border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800">
                    <th className="py-1 text-left">True \ Pred</th>
                    {routerCmLabels.map((lbl) => (
                      <th key={lbl} className="py-1 px-1 font-mono">
                        {lbl.replace("skin_", "")}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {routerCmMatrix.map((row, rIdx) => (
                    <tr key={routerCmLabels[rIdx] ?? rIdx} className="border-b border-slate-100 dark:border-slate-800/70">
                      <td className="py-1 text-left font-mono font-medium">
                        {routerCmLabels[rIdx]}
                      </td>
                      {row.map((val, cIdx) => (
                        <td
                          key={cIdx}
                          className={`py-1 px-1 font-mono ${
                            rIdx === cIdx ? "font-bold text-emerald-700 dark:text-emerald-400" : ""
                          }`}
                        >
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h2 className="text-sm font-semibold">{t("visionBench.ablationTitle")}</h2>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            {ablations.length > 0 ? (
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800">
                    <th className="py-1.5 pr-2">Variant</th>
                    <th className="py-1.5 px-2">Test AUROC</th>
                    <th className="py-1.5 px-2">Δ AUROC</th>
                    <th className="py-1.5 px-2">Dx Macro-F1</th>
                    <th className="py-1.5 pl-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {ablations.map((row) => (
                    <tr key={String(row.variant)} className="border-b border-slate-100 dark:border-slate-800/70">
                      <td className="py-1.5 pr-2 font-mono">{String(row.variant)}</td>
                      <td className="py-1.5 px-2">{Number(row.locked_test_auroc ?? 0).toFixed(4)}</td>
                      <td className="py-1.5 px-2 font-mono">{Number(row.delta_auroc_vs_full ?? 0).toFixed(4)}</td>
                      <td className="py-1.5 px-2">{Number(row.locked_test_diagnosis_macro_f1 ?? 0).toFixed(4)}</td>
                      <td className="py-1.5 pl-2">{String(row.training_seconds ?? "-")}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800">
                    <th className="py-1.5 pr-2">Comparator</th>
                    <th className="py-1.5 px-2">Comp AUROC</th>
                    <th className="py-1.5 px-2">Δ AUROC</th>
                    <th className="py-1.5 pl-2">Holm p</th>
                  </tr>
                </thead>
                <tbody>
                  {pairedBootstrap.map((row) => (
                    <tr key={row.comparator_model} className="border-b border-slate-100 dark:border-slate-800/70">
                      <td className="py-1.5 pr-2 font-mono">{row.comparator_model}</td>
                      <td className="py-1.5 px-2">{row.comparator_auroc.toFixed(4)}</td>
                      <td className="py-1.5 px-2 font-mono">+{row.delta_auroc.toFixed(4)}</td>
                      <td className="py-1.5 pl-2 font-mono">{row.p_value_holm.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
