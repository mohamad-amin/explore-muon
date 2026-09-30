// Refined search: equal-loudness optimizer palette (small lightness spread), chroma >= 0.12,
// all-pairs CVD separation (protan/deutan, Machado 2009) and normal-vision floor, contrast >= 3:1.
// usage: node palette_refine.mjs light|dark SURFACE  -> prints top candidates with OKLCH
import { readFileSync } from "node:fs";
const src = readFileSync(new URL("./palette_search.mjs", import.meta.url), "utf8");
const mode = process.argv[2], surface = process.argv[3];
// reuse the math from palette_search.mjs (everything before "const BAND")
const math = src.slice(src.indexOf("const s2lin"), src.indexOf("const BAND"));
const api = new Function("surface", math + "; return { oklchToHex, contrast, labs, d, oklabFromLin, hex2lin };")(surface);
const { oklchToHex, contrast, labs, d, oklabFromLin, hex2lin } = api;
const toLch = (h) => { const [L, a, b] = oklabFromLin(hex2lin(h)); return [L, Math.hypot(a, b), ((Math.atan2(b, a) * 180) / Math.PI + 360) % 360]; };
const LR = mode === "light" ? [0.50, 0.64] : [0.58, 0.67];
const fam = { Muon: [240, 262], PD: [30, 48], SOAP: [140, 165], SPD: [290, 315], TS: process.argv[4] === "magenta" ? [345, 368] : [70, 95] };
const pools = Object.values(fam).map(([h0, h1]) => { const out = [];
  for (let L = LR[0]; L <= LR[1] + 1e-9; L += 0.01) for (let C = 0.12; C <= 0.24; C += 0.01) for (let H = h0; H <= h1; H += 2) {
    const hex = oklchToHex(L, C, H % 360); if (hex && contrast(hex, surface) >= 3) out.push(hex); }
  return out; });
function evalPal(pal) {
  let cvd = 99, nrm = 99;
  for (let i = 0; i < pal.length; i++) for (let j = i + 1; j < pal.length; j++) {
    const A = labs(pal[i]), B = labs(pal[j]); nrm = Math.min(nrm, d(A[0], B[0])); cvd = Math.min(cvd, d(A[1], B[1]), d(A[2], B[2])); }
  const Ls = pal.map((h) => toLch(h)[0]), spread = Math.max(...Ls) - Math.min(...Ls);
  const cmin = Math.min(...pal.map((h) => toLch(h)[1]));
  return { cvd, nrm, spread, cmin, val: Math.min(cvd, 11) + 0.03 * Math.min(nrm, 30) - 25 * Math.max(0, spread - 0.08) + 4 * cmin };
}
const results = [];
for (let restart = 0; restart < 60; restart++) {
  let best = null;
  for (let it = 0; it < 20000; it++) { const pal = pools.map((p) => p[(Math.random() * p.length) | 0]); const s = evalPal(pal);
    if (s.nrm >= 18 && (!best || s.val > best.val)) best = { pal, ...s }; }
  for (let r = 0; best && r < 4; r++) for (let k = 0; k < 5; k++) for (const c of pools[k]) {
    const pal = best.pal.slice(); pal[k] = c; const s = evalPal(pal); if (s.nrm >= 18 && s.val > best.val) best = { pal, ...s }; }
  if (best) results.push(best);
}
results.sort((a, b) => b.val - a.val);
const seen = new Set();
for (const r of results) { const key = r.pal.join(); if (seen.has(key)) continue; seen.add(key);
  console.log(r.pal.join(","), `cvd ${r.cvd.toFixed(1)} normal ${r.nrm.toFixed(1)} Lspread ${r.spread.toFixed(3)} Cmin ${r.cmin.toFixed(3)}`,
    r.pal.map((h) => toLch(h).map((v, i) => v.toFixed(i === 2 ? 0 : 2)).join("/")).join("  "));
  if (seen.size >= 6) break; }
