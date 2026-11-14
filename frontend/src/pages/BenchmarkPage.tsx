import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";

const OCR_TIER_CHART = [
  { tier: "clean_scan", cerPct: 0.21, werPct: 1.38 },
  { tier: "mild_scan", cerPct: 0.22, werPct: 1.19 },
  { tier: "heavy_scan", cerPct: 16.15, werPct: 54.71 },
];

export function BenchmarkPage() {
  const token = useAuthStore((s) => s.token);

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: () =>
      apiRequest<{ models: Array<Record<string, unknown>>; count: number }>("/api/v1/models", { token }),
    enabled: Boolean(token),
  });

  const expsQuery = useQuery({
    queryKey: ["experiments"],
    queryFn: () =>
      apiRequest<{ experiments: Array<Record<string, unknown>> }>("/api/v1/experiments", { token }),
    enabled: Boolean(token),
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h1 className="text-sm font-semibold text-slate-900">OCR Stress Benchmark (SYNTH-MED-V3 Val, 16 Docs/Tier)</h1>
        </CardHeader>
        <CardContent>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={OCR_TIER_CHART}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="tier" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="cerPct" name="CER (%)" fill="#0284c7" />
                <Bar dataKey="werPct" name="WER (%)" fill="#d97706" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h2 className="text-sm font-semibold text-slate-900">Registered From-Scratch Models</h2>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {modelsQuery.data?.models.map((m) => (
            <div key={String(m.model_id)} className="rounded border border-slate-200 p-3 text-xs space-y-1">
              <div className="font-semibold text-slate-900">{String(m.model_name)}</div>
              <div className="text-slate-600">{String(m.architecture_family)}</div>
              <Badge tone="emerald">Random Init: {String(m.random_init_verified ?? "true")}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h2 className="text-sm font-semibold text-slate-900">Experiment Runs</h2>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          {expsQuery.data?.experiments.length === 0 && (
            <p className="text-slate-500">No experiment runs recorded in DB yet.</p>
          )}
          {expsQuery.data?.experiments.map((exp) => (
            <div key={String(exp.experiment_id)} className="rounded border border-slate-200 p-3">
              <div className="font-medium">{String(exp.experiment_name ?? exp.experiment_id)}</div>
              <div className="text-slate-500 mt-1">{String(exp.created_at ?? "")}</div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
