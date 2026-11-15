import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { apiRequest } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { DocumentDetailResponse, DocumentSummary, ExtractedEntity, ExtractedRelation } from "@/types/api";
import { getEntityText, getEntityTypeStyle } from "@/utils/entityStyles";

export function RelationGraphPage() {
  const navigate = useNavigate();
  const token = useAuthStore((s) => s.token);
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [selectedRel, setSelectedRel] = useState<ExtractedRelation | null>(null);

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

  const entities = detailQuery.data?.entities.slice(0, 18) ?? [];
  const entityMap = new Map<string, ExtractedEntity>(entities.map((e) => [e.entity_id, e]));
  const relations = (detailQuery.data?.relations ?? []).filter(
    (r) => entityMap.has(r.source_entity_id) && entityMap.has(r.target_entity_id)
  );

  // Circular layout in SVG (680x360)
  const cx = 340;
  const cy = 180;
  const rx = 240;
  const ry = 130;
  const positions = new Map<string, { x: number; y: number }>();
  entities.forEach((ent, idx) => {
    const angle = (2 * Math.PI * idx) / Math.max(1, entities.length);
    positions.set(ent.entity_id, {
      x: cx + rx * Math.cos(angle),
      y: cy + ry * Math.sin(angle),
    });
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-semibold text-slate-900">Clinical Relation Graph</h1>
            <p className="text-xs text-slate-500">Directed semantic links between extracted clinical entities</p>
          </div>
          <select
            aria-label="Select document"
            value={activeDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            className="rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-800"
          >
            {docsQuery.data?.documents.map((d) => (
              <option key={d.document_id} value={d.document_id}>
                {d.original_filename || d.document_id} ({d.detected_language.toUpperCase()})
              </option>
            ))}
          </select>
        </CardHeader>
        <CardContent>
          {entities.length === 0 ? (
            <p className="text-xs text-slate-500 py-8 text-center">No entities available for relation graph.</p>
          ) : (
            <div className="overflow-x-auto border border-slate-200 dark:border-slate-700 rounded bg-slate-50 dark:bg-slate-900">
              <svg viewBox="0 0 680 360" className="w-full max-w-3xl mx-auto h-96" role="img" aria-label="Relation graph">
                <defs>
                  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L8,3 z" fill="#475569" />
                  </marker>
                </defs>
                {relations.map((rel) => {
                  const p1 = positions.get(rel.source_entity_id);
                  const p2 = positions.get(rel.target_entity_id);
                  if (!p1 || !p2) return null;
                  const mx = (p1.x + p2.x) / 2;
                  const my = (p1.y + p2.y) / 2;
                  const active = selectedRel?.relation_id === rel.relation_id;
                  return (
                    <g key={rel.relation_id} data-testid="graph-edge" onClick={() => setSelectedRel(rel)} className="cursor-pointer">
                      <line
                        x1={p1.x}
                        y1={p1.y}
                        x2={p2.x}
                        y2={p2.y}
                        stroke={active ? "#0284c7" : "#64748b"}
                        strokeWidth={active ? 2.5 : 1.5}
                        markerEnd="url(#arrow)"
                      />
                      <text x={mx} y={my - 4} fontSize="9" textAnchor="middle" fill="currentColor" className="font-semibold text-slate-900 dark:text-slate-100">
                        {rel.relation_type}
                      </text>
                    </g>
                  );
                })}
                {entities.map((ent) => {
                  const p = positions.get(ent.entity_id);
                  if (!p) return null;
                  const fill = getEntityTypeStyle(ent.entity_type).hex;
                  const label = getEntityText(ent).slice(0, 18);
                  return (
                    <g
                      key={ent.entity_id}
                      data-testid="graph-node"
                      className="cursor-pointer"
                      onClick={() => navigate(`/?doc=${encodeURIComponent(activeDocId)}&entity=${encodeURIComponent(ent.entity_id)}`)}
                    >
                      <title>{`${ent.entity_type}: ${getEntityText(ent)} (click to jump to source span)`}</title>
                      <circle cx={p.x} cy={p.y} r={12} fill={fill} stroke="#ffffff" strokeWidth={2} />
                      <text x={p.x} y={p.y + 24} fontSize="10" textAnchor="middle" fill="currentColor" className="text-slate-900 dark:text-slate-100">
                        {label}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}

          {selectedRel && (
            <div className="mt-4 rounded border border-sky-200 dark:border-sky-800 bg-sky-50 dark:bg-sky-950/40 p-3 text-xs flex items-center justify-between">
              <div>
                <span className="font-semibold">{getEntityText(entityMap.get(selectedRel.source_entity_id))}</span>
                {" —["}
                <Badge tone="sky">{selectedRel.relation_type}</Badge>
                {"]→ "}
                <span className="font-semibold">{getEntityText(entityMap.get(selectedRel.target_entity_id))}</span>
              </div>
              <span className="text-slate-600 dark:text-slate-300">Confidence: {(selectedRel.confidence * 100).toFixed(1)}%</span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
