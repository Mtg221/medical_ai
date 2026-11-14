import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { AuditVerifyResult, TimelineEvent } from "@/types/api";

export function TimelineAuditPage() {
  const token = useAuthStore((s) => s.token);
  const [subjectId, setSubjectId] = useState("subj_synth_001");

  const timelineQuery = useQuery({
    queryKey: ["timeline", subjectId],
    queryFn: () =>
      apiRequest<{ timeline_events: TimelineEvent[]; contradictions_count: number }>(
        `/api/v1/subjects/${subjectId}/timeline`,
        { token }
      ),
    enabled: Boolean(token && subjectId.trim()),
    retry: false,
  });

  const auditQuery = useQuery({
    queryKey: ["audit"],
    queryFn: () => apiRequest<AuditVerifyResult>("/api/v1/audit/verify", { token }),
    enabled: Boolean(token),
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader className="flex items-center justify-between">
          <h1 className="text-sm font-semibold text-slate-900">Longitudinal Subject Timeline</h1>
          <input
            aria-label="Subject ID"
            value={subjectId}
            onChange={(e) => setSubjectId(e.target.value)}
            className="rounded border border-slate-300 px-2 py-1 text-xs"
          />
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          {timelineQuery.isError && (
            <p className="text-slate-500">Subject not found or no timeline events yet.</p>
          )}
          {timelineQuery.data?.timeline_events.map((ev) => (
            <div key={ev.event_id} className="rounded border border-slate-200 p-2.5">
              <div className="flex items-center justify-between">
                <span className="font-mono font-medium">{ev.event_date_iso}</span>
                <Badge tone={ev.contradiction_flag ? "rose" : "slate"}>{ev.event_type}</Badge>
              </div>
              <p className="mt-1 text-slate-700">{ev.summary_text}</p>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-emerald-700" aria-hidden="true" />
          <h2 className="text-sm font-semibold text-slate-900">Tamper-Evident Audit Chain</h2>
        </CardHeader>
        <CardContent className="text-xs space-y-2">
          {auditQuery.data ? (
            <div className="rounded border border-slate-200 p-3 space-y-1">
              <div>
                Chain Valid:{" "}
                <Badge tone={auditQuery.data.valid ? "emerald" : "rose"}>
                  {String(auditQuery.data.valid)}
                </Badge>
              </div>
              <div>Entries Verified: {auditQuery.data.entries_checked}</div>
            </div>
          ) : (
            <p className="text-slate-500">Checking audit chain...</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
