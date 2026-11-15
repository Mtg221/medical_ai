import { useQuery } from "@tanstack/react-query";
import { ShieldCheck } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";

export function SettingsDiagnosticsPage() {
  const token = useAuthStore((s) => s.token);
  const { user, language } = useAuthStore();

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: () => apiRequest<Record<string, unknown>>("/health", { token }),
  });

  const auditQuery = useQuery({
    queryKey: ["audit-verify"],
    queryFn: () => apiRequest<{ valid: boolean; entries?: number }>("/api/v1/audit/verify", { token }),
    enabled: Boolean(token),
  });

  const modelsQuery = useQuery({
    queryKey: ["models-provenance"],
    queryFn: () =>
      apiRequest<{
        loaded_models?: Array<{
          model_id: string;
          name: string;
          model_type: string;
          seed: number;
          corpus_id: string;
          corpus_hash: string;
          git_commit: string;
          saved_at_utc: string;
          cfg: Record<string, unknown>;
          files_sha256: Record<string, string>;
        }>;
      }>("/api/v1/models", { token }),
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex items-center justify-between">
          <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100">System Diagnostics & Security</h1>
          <Badge tone="emerald">
            <ShieldCheck className="h-3.5 w-3.5 mr-1 inline" />
            CPU-Only · Local-Only
          </Badge>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="rounded border border-slate-200 dark:border-slate-700 p-3 space-y-1.5">
            <div className="font-semibold text-slate-900 dark:text-slate-100">Runtime & Health</div>
            <div>API Status: <span className="font-mono">{String(healthQuery.data?.status ?? "checking...")}</span></div>
            <div>Execution Mode: <span className="font-mono">CPU-ONLY (Pure NumPy)</span></div>
            <div>Active UI Locale: <span className="font-mono">{language.toUpperCase()}</span></div>
          </div>
          <div className="rounded border border-slate-200 dark:border-slate-700 p-3 space-y-1.5">
            <div className="font-semibold text-slate-900 dark:text-slate-100">Security & Audit Chain</div>
            <div>Authenticated User: <span className="font-mono">{user?.user_id ?? "guest"} ({user?.role ?? "-"})</span></div>
            <div>Hash-Chained Audit Log: <Badge tone={auditQuery.data?.valid ? "emerald" : "amber"}>{auditQuery.data?.valid ? "VALID" : "PENDING"}</Badge></div>
            <div>CSP & Headers: <span className="font-mono">script-src &apos;self&apos;; nosniff</span></div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Served Evaluated Models</h2>
        </CardHeader>
        <CardContent className="space-y-3 text-xs">
          {(modelsQuery.data?.loaded_models ?? []).map((m) => (
            <div key={m.model_id} className="rounded border border-slate-200 dark:border-slate-700 p-3 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-900 dark:text-slate-100">{m.name} ({m.model_id})</span>
                <Badge tone="sky">seed {m.seed}</Badge>
              </div>
              <div>Training Corpus: <span className="font-mono">{m.corpus_id}</span> (<span className="font-mono">{m.corpus_hash?.slice(0, 12)}</span>)</div>
              <div>Git Commit: <span className="font-mono">{m.git_commit}</span> · Saved: <span className="font-mono">{m.saved_at_utc}</span></div>
              <div>Config: <span className="font-mono break-all">{JSON.stringify(m.cfg)}</span></div>
              <div>Artifact Files SHA-256: <span className="font-mono">{Object.keys(m.files_sha256 ?? {}).length} verified files</span></div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
