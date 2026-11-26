import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";

const STAGE_LATENCY_SAMPLE = [
  { stage: "Stage 1 Doc", ms: 2.1 },
  { stage: "Stage 2 BiLSTM-CRF+Rules", ms: 14.8 },
  { stage: "Stage 3 Context+RelCNN", ms: 8.4 },
  { stage: "Stage 4 Temporal CRF", ms: 3.2 },
  { stage: "Stage 5 Text Transformer", ms: 5.7 },
];

export function AblationPage() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h1 className="text-sm font-semibold text-slate-900">Hybrid Pipeline Stage Ablation & Latency</h1>
          <p className="text-xs text-slate-500">
            Stage-level architecture profile on SYNTH-MED-V3 (locked test ablation metrics will populate after Step 8)
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={STAGE_LATENCY_SAMPLE}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="stage" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="ms" name="Mean Stage Latency (ms/doc)" fill="#0284c7" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="rounded border border-slate-200 p-3">
              <div className="font-semibold text-slate-900">Stage 3 (F4 Fix)</div>
              <p className="text-slate-600 mt-1">Balanced non-present assertion BiLSTM fused with ConText + type-masked Relation CNN.</p>
            </div>
            <div className="rounded border border-slate-200 p-3">
              <div className="font-semibold text-slate-900">Stage 5 (F5 Fix)</div>
              <p className="text-slate-600 mt-1">Text-reading Transformer reranker over span + left/right context tokens + 32-dim char morphology.</p>
            </div>
            <div className="rounded border border-slate-200 p-3">
              <div className="font-semibold text-slate-900">Ablation Toggles</div>
              <p className="text-slate-600 mt-1">Supports disabling rules, assertion BiLSTM, relation CNN, temporal CRF, or Transformer fusion.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
