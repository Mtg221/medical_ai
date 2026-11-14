import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { BenchmarkPage } from "@/pages/BenchmarkPage";
import { LoginPage } from "@/pages/LoginPage";
import { TimelineAuditPage } from "@/pages/TimelineAuditPage";
import { UploadJobsPage } from "@/pages/UploadJobsPage";
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
              <>
                <Route path="*" element={<LoginPage />} />
              </>
            ) : (
              <>
                <Route index element={<WorkspacePage />} />
                <Route path="upload" element={<UploadJobsPage />} />
                <Route path="benchmark" element={<BenchmarkPage />} />
                <Route path="timeline" element={<TimelineAuditPage />} />
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
