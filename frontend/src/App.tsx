import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { AblationPage } from "@/pages/AblationPage";
import { BenchmarkPage } from "@/pages/BenchmarkPage";
import { CorpusExplorerPage } from "@/pages/CorpusExplorerPage";
import { ErrorTaxonomyPage } from "@/pages/ErrorTaxonomyPage";
import { LoginPage } from "@/pages/LoginPage";
import { PdfViewerPage } from "@/pages/PdfViewerPage";
import { RelationGraphPage } from "@/pages/RelationGraphPage";
import { ReportsPage } from "@/pages/ReportsPage";
import { ReviewQueuePage } from "@/pages/ReviewQueuePage";
import { SchemaExplorerPage } from "@/pages/SchemaExplorerPage";
import { SettingsDiagnosticsPage } from "@/pages/SettingsDiagnosticsPage";
import { SkinVisionPage } from "@/pages/SkinVisionPage";
import { TimelineAuditPage } from "@/pages/TimelineAuditPage";
import { UploadJobsPage } from "@/pages/UploadJobsPage";
import { VisionBenchmarkPage } from "@/pages/VisionBenchmarkPage";
import { WorkspacePage } from "@/pages/WorkspacePage";
import { useAuthStore } from "@/store/authStore";

const queryClient = new QueryClient();

export function App() {
  const token = useAuthStore((s) => s.token);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/react">
        <Routes>
          <Route element={<Layout />}>
            {!token ? (
              <Route path="*" element={<LoginPage />} />
            ) : (
              <>
                <Route index element={<WorkspacePage />} />
                <Route path="upload" element={<UploadJobsPage />} />
                <Route path="skin" element={<SkinVisionPage />} />
                <Route path="vision-benchmark" element={<VisionBenchmarkPage />} />
                <Route path="pdf" element={<PdfViewerPage />} />
                <Route path="relations" element={<RelationGraphPage />} />
                <Route path="review" element={<ReviewQueuePage />} />
                <Route path="benchmark" element={<BenchmarkPage />} />
                <Route path="ablation" element={<AblationPage />} />
                <Route path="errors" element={<ErrorTaxonomyPage />} />
                <Route path="corpus" element={<CorpusExplorerPage />} />
                <Route path="reports" element={<ReportsPage />} />
                <Route path="timeline" element={<TimelineAuditPage />} />
                <Route path="schema" element={<SchemaExplorerPage />} />
                <Route path="settings" element={<SettingsDiagnosticsPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </>
            )}
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
