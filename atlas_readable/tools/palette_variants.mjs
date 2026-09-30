// Palette for arms that are NOT optimizers (e.g. coordination variants of PD): 4 colors, all-pairs CVD-safe among
// themselves, contrast >= 3:1, and as far as possible from the optimizer identity colors used across the Atlas.
// usage: node palette_variants.mjs light|dark SURFACE
import { readFileSync } from "node:fs";
const src = readFileSync(new URL("./palette_search.mjs", import.meta.url), "utf8");
const mode = process.argv[2], surface = process.argv[3];
const math = src.slice(src.indexOf("const s2lin"), src.indexOf("const BAND"));
const { oklchToHex, contrast, labs, d } = new Function("surface", math + "; return { oklchToHex, contrast, labs, d };")(surface);
const OPT = mode === "light" ? ["#0c8dd9", "#ab4c12", "#33a463", "#7543dd", "#c40a8f", "#b98b00"] : ["#129fe7", "#c35408", "#0ab074", "#7f52f5", "#c93196", "#d4a520"];
const LR = mode === "light" ? [0.44, 0.66] : [0.52, 0.67];
// one hue family per slot, the same in both themes: indigo, coral, olive, lilac
const HUES = [[255, 280], [12, 32], [100, 132], [298, 328]];
const far = (h) => Math.min(...OPT.map((o) => d(labs(h)[0], labs(o)[0])));
const pools = HUES.map(([h0, h1]) => { const out = [];
  for (let L = LR[0]; L <= LR[1] + 1e-9; L += 0.02) for (let C = 0.1; C <= 0.24; C += 0.01) for (let H = h0; H <= h1; H += 2) {
    const hex = oklchToHex(L, C, H % 360); if (hex && contrast(hex, surface) >= 3 && far(hex) >= 10) out.push(hex); }
  return out; });
const cand = pools.flat();
function score(p) {
  let cvd = 99, nrm = 99;
  for (let i = 0; i < p.length; i++) for (let j = i + 1; j < p.length; j++) { const A = labs(p[i]), B = labs(p[j]);
    nrm = Math.min(nrm, d(A[0], B[0])); cvd = Math.min(cvd, d(A[1], B[1]), d(A[2], B[2])); }
  return { cvd, nrm, far: Math.min(...p.map(far)), val: Math.min(cvd, 10) + 0.05 * Math.min(nrm, 30) + 0.3 * Math.min(...p.map(far)) };
}
let best = null;
for (let r = 0; r < 40; r++) {
  let cur = null;
  for (let it = 0; it < 30000; it++) { const p = pools.map((pl) => pl[(Math.random() * pl.length) | 0]); const s = score(p);
    if (s.nrm >= 18 && s.cvd >= 8 && (!cur || s.val > cur.val)) cur = { p, ...s }; }
  for (let k = 0; cur && k < 3; k++) for (let i = 0; i < 4; i++) for (const c of pools[i]) { const p = cur.p.slice(); p[i] = c; const s = score(p);
    if (s.nrm >= 18 && s.cvd >= 8 && s.val > cur.val) cur = { p, ...s }; }
  if (cur && (!best || cur.val > best.val)) best = cur;
}
console.log(mode, best ? `${best.p.join(",")}  cvd ${best.cvd.toFixed(1)} normal ${best.nrm.toFixed(1)} min distance to optimizer colors ${best.far.toFixed(1)}` : "none", `(pool ${cand.length})`);
