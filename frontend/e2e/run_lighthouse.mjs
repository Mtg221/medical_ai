import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_DIR = path.resolve(__dirname, "..");
const REPO_DIR = path.resolve(FRONTEND_DIR, "..");

const BASE = process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:8011";
const CHROME =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ||
  (process.platform === "darwin" ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "google-chrome");
const LH_BIN = path.resolve(FRONTEND_DIR, "node_modules/.bin/lighthouse");

const TARGET_PAGES = [
  { name: "login", url: `${BASE}/react/` },
  { name: "workspace", url: `${BASE}/react/` },
  { name: "upload", url: `${BASE}/react/upload` },
  { name: "review", url: `${BASE}/react/review` },
  { name: "reports", url: `${BASE}/react/reports` },
];

const results = [];
for (const target of TARGET_PAGES) {
  const outJson = path.resolve(process.cwd(), `.lighthouse_${target.name}.json`);
  execFileSync(
    LH_BIN,
    [
      target.url,
      "--only-categories=accessibility",
      "--output=json",
      `--output-path=${outJson}`,
      `--chrome-flags=--headless=new --no-sandbox --disable-gpu`,
      "--quiet",
    ],
    {
      env: { ...process.env, CHROME_PATH: CHROME },
      stdio: "inherit",
    }
  );
  const report = JSON.parse(fs.readFileSync(outJson, "utf8"));
  const score = Math.round((report.categories?.accessibility?.score ?? 0) * 100);
  const failedAudits = Object.values(report.audits || {})
    .filter((a) => a.score !== null && a.score < 1 && a.details?.items?.length > 0)
    .map((a) => ({ id: a.id, title: a.title, score: a.score, items: a.details.items.length }));
  results.push({ page: target.name, url: target.url, accessibility_score: score, failed_audits: failedAudits });
  fs.unlinkSync(outJson);
}

const outPath = path.resolve(REPO_DIR, "docs/results/lighthouse_accessibility_report.json");
fs.writeFileSync(outPath, JSON.stringify({ threshold: 95, pages: results }, null, 2));
console.log("Lighthouse summary:", JSON.stringify(results, null, 2));

const below = results.filter((r) => r.accessibility_score < 95);
if (below.length > 0) {
  console.error("Accessibility score below 95:", below);
  process.exit(1);
}
