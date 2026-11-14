import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";

export function SchemaExplorerPage() {
  const token = useAuthStore((s) => s.token);

  const schemaQuery = useQuery({
    queryKey: ["extraction-schema"],
    queryFn: () => apiRequest<Record<string, unknown>>("/api/v1/schema/extraction", { token }),
    enabled: Boolean(token),
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex items-center justify-between">
          <div>
            <h1 className="text-sm font-semibold text-slate-900">Extraction Output Contract (JSON Schema)</h1>
            <p className="text-xs text-slate-500">Single source of truth served from /api/v1/schema/extraction (Pydantic v2)</p>
          </div>
          <Badge tone="sky">schema_version: 1.0</Badge>
        </CardHeader>
        <CardContent>
          <pre className="rounded border border-slate-200 bg-slate-900 text-slate-100 p-4 text-[11px] font-mono overflow-auto max-h-[540px]">
            {JSON.stringify(schemaQuery.data ?? { status: "Loading schema..." }, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
