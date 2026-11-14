import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

const SPLIT_STATS = [
  { split: "train", docs: 600, vocab: 3221, sentOverlapPct: 100, entOverlapPct: 100 },
  { split: "val", docs: 160, vocab: 2211, sentOverlapPct: 11.3, entOverlapPct: 73.0 },
  { split: "test_iid", docs: 160, vocab: 2253, sentOverlapPct: 12.0, entOverlapPct: 73.0 },
  { split: "test_unseen_terms", docs: 160, vocab: 1809, sentOverlapPct: 8.3, entOverlapPct: 20.8 },
  { split: "test_unseen_templates", docs: 160, vocab: 1955, sentOverlapPct: 0.6, entOverlapPct: 73.6 },
];

export function CorpusExplorerPage() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h1 className="text-sm font-semibold text-slate-900">SYNTH-MED-V3 Corpus Explorer (1,240 Docs)</h1>
            <p className="text-xs text-slate-500">Bilingual EN/FR generalization splits with 0 doubled spans and 100% offset integrity</p>
          </div>
          <Badge tone="emerald">Audit Passed (0 Doubled)</Badge>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={SPLIT_STATS.filter((s) => s.split !== "train")}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="split" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="sentOverlapPct" name="Masked Sentence Overlap vs Train (%)" fill="#0284c7" />
                <Bar dataKey="entOverlapPct" name="Unique Entity Overlap vs Train (%)" fill="#0d9488" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-600">
                  <th className="py-2 pr-4">Split</th>
                  <th className="py-2 pr-4">Docs</th>
                  <th className="py-2 pr-4">Vocab</th>
                  <th className="py-2 pr-4">Masked Sent Overlap</th>
                  <th className="py-2">Unique Entity Overlap</th>
                </tr>
              </thead>
              <tbody>
                {SPLIT_STATS.map((s) => (
                  <tr key={s.split} className="border-b border-slate-100">
                    <td className="py-2 pr-4 font-mono font-medium">{s.split}</td>
                    <td className="py-2 pr-4">{s.docs}</td>
                    <td className="py-2 pr-4">{s.vocab}</td>
                    <td className="py-2 pr-4">{s.sentOverlapPct}%</td>
                    <td className="py-2">{s.entOverlapPct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
