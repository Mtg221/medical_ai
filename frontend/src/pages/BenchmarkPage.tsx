import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";

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
          <h1 className="text-sm font-semibold text-slate-900">Registered From-Scratch Models</h1>
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
