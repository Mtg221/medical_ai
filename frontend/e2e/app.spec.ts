import { expect, test } from "@playwright/test";
import path from "node:path";

const SCREENSHOT_DIR = path.resolve(process.cwd(), "../docs/screenshots");
const DEMO_PW = process.env.MEDRESEARCH_DEMO_PASSWORD || "e2e-demo-pw-2026";

function makeMinimalPdfBuffer(textLine: string): Buffer {
  const safe = textLine.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
  const stream = `BT\n/F1 12 Tf\n72 720 Td\n(${safe}) Tj\nET\n`;
  const objects = [
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
    `4 0 obj\n<< /Length ${Buffer.byteLength(stream, "utf8")} >>\nstream\n${stream}endstream\nendobj\n`,
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
  ];
  let pdf = "%PDF-1.4\n";
  const offsets: number[] = [0];
  for (const obj of objects) {
    offsets.push(Buffer.byteLength(pdf, "utf8"));
    pdf += obj;
  }
  const xrefOffset = Buffer.byteLength(pdf, "utf8");
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (let i = 1; i <= objects.length; i++) {
    pdf += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Buffer.from(pdf, "utf8");
}

const ALL_PAGES: Array<{ slug: string; route: string }> = [
  { slug: "workspace", route: "/react/" },
  { slug: "upload", route: "/react/upload" },
  { slug: "pdf", route: "/react/pdf" },
  { slug: "relations", route: "/react/relations" },
  { slug: "review", route: "/react/review" },
  { slug: "benchmark", route: "/react/benchmark" },
  { slug: "ablation", route: "/react/ablation" },
  { slug: "errors", route: "/react/errors" },
  { slug: "corpus", route: "/react/corpus" },
  { slug: "reports", route: "/react/reports" },
  { slug: "timeline", route: "/react/timeline" },
  { slug: "schema", route: "/react/schema" },
  { slug: "settings", route: "/react/settings" },
];

test.describe("Phase 12B Full E2E (EN + FR) & 14-Page Light/Dark Screenshots", () => {
  test("EN + FR upload -> highlights -> open fact -> review persists after reload -> report download + 28 screenshots", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);
    const tokResp = await request.post("/api/v1/auth/token", {
      data: { username: "admin_dakar", password: DEMO_PW },
    });
    expect(tokResp.ok()).toBeTruthy();
    const { access_token: token } = (await tokResp.json()) as { access_token: string };
    const seedResp = await request.post("/api/v1/experiments/run", {
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(seedResp.ok()).toBeTruthy();

    await page.setViewportSize({ width: 1440, height: 900 });

    // Capture Login page light + dark BEFORE signing in
    await page.goto("/react/");
    await expect(page.getByLabel(/username/i)).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "login_light.png"), fullPage: true });
    await page.getByRole("button", { name: "Switch to dark mode" }).click();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "login_dark.png"), fullPage: true });
    await page.getByRole("button", { name: "Switch to light mode" }).click();

    // Sign in (EN flow)
    await page.getByLabel(/username/i).fill("admin_dakar");
    await page.getByLabel(/password/i).fill(DEMO_PW);
    await page.getByRole("button", { name: /sign in/i }).click();
    await expect(page.getByRole("navigation", { name: "Primary navigation" })).toBeVisible();

    // Upload a real EN PDF
    await page.getByRole("link", { name: /upload/i }).click();
    await page.locator("#doc-file").setInputFiles({
      name: "e2e_en_note.pdf",
      mimeType: "application/pdf",
      buffer: makeMinimalPdfBuffer("Patient has acute pneumonia and takes metformin 500 mg daily."),
    });
    await page.locator("form button[type='submit']").click();
    await expect(page.getByText(/SUCCEEDED/).first()).toBeVisible({ timeout: 20_000 });

    // Workspace: verify highlights appear and click a fact
    await page.getByRole("link", { name: /workspace/i }).click();
    await page.getByRole("button", { name: /e2e_en_note\.pdf/i }).first().click();
    const firstHighlight = page.locator("pre button").first();
    await expect(firstHighlight).toBeVisible({ timeout: 10_000 });
    await firstHighlight.click();

    // Review queue: review & reload to verify persistence
    await page.getByRole("link", { name: "Review" }).click();
    await expect(page.getByRole("heading", { name: "Human-in-the-Loop Review Mode" })).toBeVisible();
    await page.keyboard.press("a");
    await expect(page.getByText(/Saved APPROVED_GROUND_TRUTH/)).toBeVisible();
    await page.reload();
    await expect(page.getByText("APPROVED_GROUND_TRUTH").first()).toBeVisible();

    // Reports: preview and download EN report
    await page.getByRole("link", { name: "Reports" }).click();
    await expect(page.getByRole("button", { name: "Download Report" })).toBeEnabled({ timeout: 10_000 });
    const [enDownload] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download Report" }).click(),
    ]);
    expect(enDownload.suggestedFilename()).toMatch(/_report_en\.md$/);

    // FR flow: toggle language to FR, upload FR PDF, verify highlights, review & download FR report
    await page.getByRole("button", { name: "Toggle language" }).click();
    await expect(page.getByRole("button", { name: "Toggle language" })).toContainText("FR");

    await page.locator("nav a[href='/react/upload']").click();
    await page.locator("#doc-file").setInputFiles({
      name: "e2e_fr_note.pdf",
      mimeType: "application/pdf",
      buffer: makeMinimalPdfBuffer("Le patient presente une pneumonie et prend metformine 500 mg."),
    });
    await page.locator("form button[type='submit']").click();
    await expect(page.getByText(/SUCCEEDED/).first()).toBeVisible({ timeout: 20_000 });

    await page.locator("nav a").first().click();
    await page.getByRole("button", { name: /e2e_fr_note\.pdf/i }).first().click();
    const dosageHighlight = page.locator("pre button[data-entity-type='DOSAGE']", { hasText: "500 mg" }).first();
    await expect(dosageHighlight).toBeVisible({ timeout: 10_000 });
    await dosageHighlight.click();
    await page.getByRole("button", { name: /Relations/i }).click();
    await expect(page.getByText(/HAS_DOSAGE/i).first()).toBeVisible({ timeout: 5_000 });
    await expect(page.getByText(/metformine/i).first()).toBeVisible();

    await page.locator("nav a[href='/react/review']").click();
    await expect(page.getByRole("heading", { name: /Review Mode|Human-in-the-Loop/i })).toBeVisible();
    await page.locator("main").click();
    await page.keyboard.press("n");
    await page.keyboard.press("c");
    await expect(page.getByLabel("Corrected entity type")).toBeVisible({ timeout: 10_000 });
    await page.getByLabel("Corrected entity type").selectOption("MEDICATION");
    await page.getByLabel("Corrected assertion status").selectOption("AFFIRMED");
    await page.getByRole("button", { name: "Save Correction" }).click();
    await expect(page.getByText(/Saved HUMAN_CORRECTED/)).toBeVisible();
    await page.reload();
    await expect(page.getByText("HUMAN_CORRECTED").first()).toBeVisible();

    await page.locator("nav a[href='/react/reports']").click();
    await page.getByRole("button", { name: "FR", exact: true }).click();
    await expect(page.getByRole("button", { name: "Download Report" })).toBeEnabled({ timeout: 10_000 });
    const [frDownload] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download Report" }).click(),
    ]);
    expect(frDownload.suggestedFilename()).toMatch(/_report_fr\.md$/);

    // Verify Relation Graph page renders >=1 node and >=1 edge and node click jumps to workspace
    await page.goto("/react/relations");
    await expect(page.locator("[data-testid='graph-node']").first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("[data-testid='graph-edge']").first()).toBeVisible({ timeout: 10_000 });
    expect(await page.locator("[data-testid='graph-node']").count()).toBeGreaterThanOrEqual(1);
    expect(await page.locator("[data-testid='graph-edge']").count()).toBeGreaterThanOrEqual(1);
    await page.locator("[data-testid='graph-node']").first().click();
    await expect(page).toHaveURL(/\/react\/?\?doc=.*&entity=/);

    // Switch back to EN and capture all remaining 13 authenticated pages x 2 themes (light + dark)
    await page.getByRole("button", { name: "Toggle language" }).click();
    for (const pg of ALL_PAGES) {
      await page.goto(pg.route);
      await page.waitForLoadState("domcontentloaded");
      await page.waitForTimeout(350);
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${pg.slug}_light.png`), fullPage: true });
    }
    await page.getByRole("button", { name: "Switch to dark mode" }).click();
    await expect(page.locator("html")).toHaveClass(/dark/);
    for (const pg of ALL_PAGES) {
      await page.goto(pg.route);
      await page.waitForLoadState("domcontentloaded");
      await page.waitForTimeout(350);
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${pg.slug}_dark.png`), fullPage: true });
    }
  });
});
