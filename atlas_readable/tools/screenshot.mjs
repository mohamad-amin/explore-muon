// Render every view in headless Chromium, capture full-page screenshots, console errors, horizontal overflow,
// and every read of the embedded payload (via a recording Proxy) that asks for a key the payload does not have.
// usage (see README): node tools/screenshot.mjs [--widths 1600,390] [--views a,b] [--dark] [--out build/screens]
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const arg = (name, def) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : def; };
const widths = arg("--widths", "1600,390").split(",").map(Number);
const dark = process.argv.includes("--dark");
const outDir = resolve(arg("--out", "build/screens"));
const pagePath = resolve(arg("--page", "gauss_newton_atlas_readable.html"));
const localPlotly = resolve("node_modules/plotly.js-dist-min/plotly.min.js");
const ALL = ["principles", "curv", "train", "gap", "spec", "gn", "lags", "coupling", "blocks", "momentum", "gap2gn", "edge", "steprule", "batch"];
const views = arg("--views", ALL.join(",")).split(",");

// installed before any page script: wraps the big JSON.parse result (the payload) in a recording proxy
const recorder = () => {
  const missing = new Map();
  window.__missingReads = missing;
  const IGNORE = new Set(["then", "toJSON", "constructor", "length", "__proto__", "valueOf", "toString", "$$typeof", "nodeType"]);
  const wrap = (obj, path) => new Proxy(obj, {
    get(t, k, r) {
      const v = Reflect.get(t, k, r);
      if (typeof k === "symbol" || IGNORE.has(k)) return v;
      if (Array.isArray(t) && (/^\d+$/.test(k) || k in Array.prototype)) return v && typeof v === "object" ? wrap(v, `${path}[]`) : v;
      if (!(k in t)) { const p = `${path}.${k}`; missing.set(p, (missing.get(p) || 0) + 1); }
      return v && typeof v === "object" ? wrap(v, Array.isArray(t) ? `${path}[]` : `${path}.${/^\d+$/.test(k) ? "<n>" : k.length > 40 ? "<key>" : k}`) : v;
    },
  });
  const parse = JSON.parse;
  JSON.parse = function (s, rev) { const v = parse.call(this, s, rev); return typeof s === "string" && s.length > 500000 && v && typeof v === "object" ? wrap(v, "D") : v; };
};

const browser = await chromium.launch();
const report = [];
for (const width of widths) {
  const ctx = await browser.newContext({ viewport: { width, height: width < 700 ? 844 : 1000 }, deviceScaleFactor: 1, colorScheme: dark ? "dark" : "light" });
  await ctx.route("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js", (route) => route.fulfill({ path: localPlotly, contentType: "application/javascript" }));
  await ctx.addInitScript(recorder);
  mkdirSync(`${outDir}/${width}${dark ? "-dark" : ""}`, { recursive: true });
  for (const view of views) {
    const page = await ctx.newPage();
    const errors = [];
    page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(`${m.type()}: ${m.text()}`); });
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    const t0 = Date.now();
    await page.goto(`file://${pagePath}#${view}`, { waitUntil: "load" });
    await page.waitForFunction(() => document.querySelector("#view-title") && document.querySelector("#view-title").textContent.length > 0, null, { timeout: 20000 });
    const firstPaint = Date.now() - t0;
    await page.evaluate(() => window.__atlasDrawAll && window.__atlasDrawAll());
    await page.waitForFunction(() => [...document.querySelectorAll(".plot")].every((p) => p.querySelector(".main-svg") || p.textContent.length), null, { timeout: 60000 }).catch(() => errors.push("timeout waiting for plots"));
    await page.waitForTimeout(400);
    const info = await page.evaluate(() => ({
      title: document.querySelector("#view-title").textContent,
      panels: document.querySelectorAll(".plot").length,
      plotted: document.querySelectorAll(".plot .main-svg").length,
      tables: document.querySelectorAll("table.data").length,
      empties: [...document.querySelectorAll(".empty, .note")].map((e) => e.textContent.slice(0, 120)),
      overflowX: document.documentElement.scrollWidth - window.innerWidth,
      height: document.documentElement.scrollHeight,
      plotSizes: [...document.querySelectorAll(".plot")].map((p) => [Math.round(p.getBoundingClientRect().width), Math.round(p.getBoundingClientRect().height)]),
      missing: [...(window.__missingReads || new Map()).entries()],
    }));
    // grow the viewport to the whole page (instead of fullPage capture, whose resize makes Plotly redraw mid-shot)
    await page.setViewportSize({ width, height: Math.min(info.height + 40, 30000) });
    await page.waitForTimeout(1500);
    const file = `${outDir}/${width}${dark ? "-dark" : ""}/${view}.png`;
    await page.screenshot({ path: file, fullPage: false });
    await page.setViewportSize({ width, height: width < 700 ? 844 : 1000 });
    report.push({ width, view, firstPaintMs: firstPaint, ...info, errors });
    console.log(`${String(width).padStart(4)} ${view.padEnd(11)} panels ${info.plotted}/${info.panels} tables ${info.tables} overflowX ${info.overflowX} height ${info.height} ` +
      `plot sizes ${[...new Set(info.plotSizes.map((s) => s.join("x")))].join(" ")} errors ${errors.length} missing-reads ${info.missing.length}`);
    errors.forEach((e) => console.log("      ", e));
    await page.close();
  }
  await ctx.close();
}
writeFileSync(`${outDir}/report${dark ? "-dark" : ""}.json`, JSON.stringify(report, null, 1));
await browser.close();
