import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";

const ERROR_CATEGORIES = [
  { category: "assertion_wrong", count: 604, note: "Scope termination / double negation / distant cues" },
  { category: "relation_wrong", count: 522, note: "Multi-entity proximity / cross-clause pairing" },
  { category: "type_confusion", count: 94, note: "SYMPTOM vs DIAGNOSIS / PROCEDURE overlap" },
  { category: "boundary_error", count: 64, note: "Multi-word modifier or hyphenated spans" },
  { category: "spurious_fp", count: 23, note: "False positive span outside gold" },
  { category: "missed_fn", count: 16, note: "Unseen morphological variant missed" },
];

export function ErrorTaxonomyPage() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h1 className="text-sm font-semibold text-slate-900">Error Taxonomy (SYNTH-MED-V3 Val — BiLSTM-CRF)</h1>
          <p className="text-xs text-slate-500">Observed error distribution from Phase 11 Step 3 validation run (160 docs, 3 seeds)</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ERROR_CATEGORIES}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Error Count" fill="#dc2626" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2 text-xs">
            {ERROR_CATEGORIES.map((row) => (
              <div key={row.category} className="rounded border border-slate-200 p-2.5 flex items-center justify-between">
                <div>
                  <span className="font-semibold text-slate-900">{row.category}</span>
                  <span className="text-slate-500 ml-2">— {row.note}</span>
                </div>
                <span className="font-mono font-semibold text-slate-800">{row.count}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
