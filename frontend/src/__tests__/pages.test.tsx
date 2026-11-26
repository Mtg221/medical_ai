import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeAll, describe, expect, it, vi } from "vitest";
import i18n from "@/i18n";
import { AblationPage } from "@/pages/AblationPage";
import { CorpusExplorerPage } from "@/pages/CorpusExplorerPage";
import { ErrorTaxonomyPage } from "@/pages/ErrorTaxonomyPage";
import { PdfViewerPage } from "@/pages/PdfViewerPage";
import { RelationGraphPage } from "@/pages/RelationGraphPage";
import { ReviewQueuePage } from "@/pages/ReviewQueuePage";
import { SettingsDiagnosticsPage } from "@/pages/SettingsDiagnosticsPage";
import { SkinVisionPage } from "@/pages/SkinVisionPage";
import { VisionBenchmarkPage } from "@/pages/VisionBenchmarkPage";

beforeAll(() => {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "ok",
        items: [],
        documents: [],
        entities: [],
        relations: [],
      }),
    })),
  );
});

function renderWithProviders(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Phase 11 frontend pages", () => {
  it("renders CorpusExplorerPage with SYNTH-MED-V3 split summary", () => {
    renderWithProviders(<CorpusExplorerPage />);
    expect(screen.getByText(/SYNTH-MED-V3/i)).toBeTruthy();
  });

  it("renders ErrorTaxonomyPage", () => {
    renderWithProviders(<ErrorTaxonomyPage />);
    expect(document.body.textContent?.length).toBeGreaterThan(20);
  });

  it("renders AblationPage", () => {
    renderWithProviders(<AblationPage />);
    expect(document.body.textContent?.length).toBeGreaterThan(20);
  });

  it("renders PdfViewerPage", () => {
    renderWithProviders(<PdfViewerPage />);
    expect(document.body.textContent?.length).toBeGreaterThan(20);
  });

  it("renders ReviewQueuePage", () => {
    renderWithProviders(<ReviewQueuePage />);
    expect(document.body.textContent?.length).toBeGreaterThan(20);
  });

  it("renders RelationGraphPage", () => {
    renderWithProviders(<RelationGraphPage />);
    expect(document.body.textContent?.length).toBeGreaterThan(20);
  });

  it("renders SettingsDiagnosticsPage", () => {
    renderWithProviders(<SettingsDiagnosticsPage />);
    expect(document.body.textContent?.length).toBeGreaterThan(20);
  });

  it("renders SkinVisionPage", () => {
    renderWithProviders(<SkinVisionPage />);
    expect(screen.getByText(/Skin Image Analysis/i)).toBeTruthy();
  });

  it("renders VisionBenchmarkPage", () => {
    renderWithProviders(<VisionBenchmarkPage />);
    expect(screen.getByText(/Vision Benchmark/i)).toBeTruthy();
  });

  it("has complete EN and FR i18n key parity for skin and visionBench", () => {
    const enBundle = i18n.getResourceBundle("en", "translation") as Record<string, Record<string, string>>;
    const frBundle = i18n.getResourceBundle("fr", "translation") as Record<string, Record<string, string>>;
    for (const ns of ["nav", "upload", "skin", "visionBench"]) {
      expect(Object.keys(frBundle[ns]).sort()).toEqual(Object.keys(enBundle[ns]).sort());
    }
  });
});
