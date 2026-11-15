import { expect, test } from "@playwright/test";
import path from "node:path";

const SCREENSHOT_DIR = path.resolve(process.cwd(), "../docs/screenshots");
const DEMO_PW = process.env.MEDRESEARCH_DEMO_PASSWORD || "e2e-demo-pw-2026";

test.describe("Phase 11B Step 7 Real-API E2E", () => {
  test("login, review persistence (A/C shortcuts + reload), dark mode, and desktop/mobile screenshots", async ({
    page,
    request,
  }) => {
    // Seed demo corpus via real API
    const tokResp = await request.post("/api/v1/auth/token", {
      data: { username: "admin_dakar", password: DEMO_PW },
    });
    expect(tokResp.ok()).toBeTruthy();
    const { access_token: token } = (await tokResp.json()) as { access_token: string };
    const seedResp = await request.post("/api/v1/experiments/run", {
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(seedResp.ok()).toBeTruthy();

    // 1. Desktop viewport (1440x900)
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/react/");

    // Login through UI
    await page.getByLabel(/username/i).fill("admin_dakar");
    await page.getByLabel(/password/i).fill(DEMO_PW);
    await page.getByRole("button", { name: /sign in/i }).click();
    await expect(page.getByRole("navigation", { name: "Primary navigation" })).toBeVisible();

    // Navigate to Review mode
    await page.getByRole("link", { name: "Review" }).click();
    await expect(page.getByRole("heading", { name: "Human-in-the-Loop Review Mode" })).toBeVisible();
    await expect(page.getByRole("progressbar")).toBeVisible();

    // Approve first entity via keyboard shortcut A
    await page.keyboard.press("a");
    await expect(page.getByText(/Saved APPROVED_GROUND_TRUTH/)).toBeVisible();

    // Move to next entity via N and open correction form via C
    await page.keyboard.press("n");
    await page.keyboard.press("c");
    await expect(page.getByRole("button", { name: "Save Correction" })).toBeVisible();
    await page.getByLabel("Corrected entity type").selectOption("SYMPTOM");
    await page.getByLabel("Corrected assertion status").selectOption("NEGATED");
    await page.getByRole("button", { name: "Save Correction" }).click();
    await expect(page.getByText(/Saved HUMAN_CORRECTED/)).toBeVisible();

    // Reload page and confirm persisted statuses + audit log survive reload
    await page.reload();
    await expect(page.getByRole("heading", { name: "Human-in-the-Loop Review Mode" })).toBeVisible();
    await expect(page.getByText("APPROVED_GROUND_TRUTH").first()).toBeVisible();
    await expect(page.getByText("HUMAN_CORRECTED").first()).toBeVisible();
    await expect(page.getByText(/Saved Review Audit Log/)).toBeVisible();

    // Capture desktop light screenshot
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "review_desktop_light.png"), fullPage: true });

    // Toggle Dark Mode and capture desktop dark screenshot
    await page.getByRole("button", { name: "Switch to dark mode" }).click();
    await expect(page.locator("html")).toHaveClass(/dark/);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "review_desktop_dark.png"), fullPage: true });

    // 2. Mobile viewport (390x844) screenshot
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "review_mobile_dark.png"), fullPage: true });
  });
});
