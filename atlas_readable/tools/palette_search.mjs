// Search OKLCH space for an optimizer palette (Muon, PD, SOAP-Muon, S∘PD, TS) that passes the
// dataviz validator's categorical checks on ALL pairs (optimizers share every chart), in one theme.
// usage: node palette_search.mjs light|dark SURFACE_HEX
// The winner is re-checked with the dataviz skill's validate_palette.js CLI afterwards.
const mode = process.argv[2] || "light";
const surface = process.argv[3] || (mode === "light" ? "#ffffff" : "#161d24");

const s2lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const lin2s = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const hex2lin = (h) => [0, 2, 4].map((i) => s2lin(parseInt(h.slice(1 + i, 3 + i), 16) / 255));
const relLum = (h) => { const [r, g, b] = hex2lin(h); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
const contrast = (a, b) => { const [hi, lo] = [relLum(a), relLum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };
function oklabFromLin([r, g, b]) {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function oklchToHex(L, C, H) {
  const a = C * Math.cos((H * Math.PI) / 180), b = C * Math.sin((H * Math.PI) / 180);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b, m_ = L - 0.1055613458 * a - 0.0638541728 * b, s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  if (rgb.some((v) => v < -1e-4 || v > 1 + 1e-4)) return null; // out of sRGB gamut
  return "#" + rgb.map((v) => Math.round(Math.max(0, Math.min(1, lin2s(v))) * 255).toString(16).padStart(2, "0")).join("");
}
const MACHADO = {
  protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
};
const sim = (lin, M) => M.map((row) => Math.max(0, Math.min(1, row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2])));
const cache = new Map();
function labs(h) {
  if (!cache.has(h)) { const l = hex2lin(h); cache.set(h, [oklabFromLin(l), oklabFromLin(sim(l, MACHADO.protan)), oklabFromLin(sim(l, MACHADO.deutan))]); }
  return cache.get(h);
}
const d = (p, q) => 100 * Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

const BAND = { light: [0.45, 0.66], dark: [0.52, 0.67] };
const families = {
  Muon: [238, 266], PD: [28, 50], SOAP: [138, 168], SPD: [288, 322], TS: null,
};
const fifth = { gold: [72, 100], magenta: [340, 372], cyan: [195, 218] };
function candidates([h0, h1]) {
  const out = [];
  for (let L = BAND[mode][0]; L <= BAND[mode][1] + 1e-9; L += 0.02)
    for (let C = 0.1; C <= 0.24; C += 0.02)
      for (let H = h0; H <= h1; H += 4) {
        const hex = oklchToHex(L, C, H % 360);
        if (hex && contrast(hex, surface) >= 3) out.push(hex);
      }
  return out;
}
function score(pal) {
  let cvd = 99, nrm = 99;
  for (let i = 0; i < pal.length; i++) for (let j = i + 1; j < pal.length; j++) {
    const A = labs(pal[i]), B = labs(pal[j]);
    nrm = Math.min(nrm, d(A[0], B[0])); cvd = Math.min(cvd, d(A[1], B[1]), d(A[2], B[2]));
  }
  return { cvd, nrm };
}
for (const [fname, range] of Object.entries(fifth)) {
  const pools = [families.Muon, families.PD, families.SOAP, families.SPD, range].map(candidates);
  let best = null;
  for (let it = 0; it < 400000; it++) {
    const pal = pools.map((p) => p[(Math.random() * p.length) | 0]);
    const s = score(pal);
    if (s.nrm < 18) continue;
    const val = Math.min(s.cvd, 12) + 0.02 * s.nrm; // prioritise CVD separation, then normal-vision spread
    if (!best || val > best.val) best = { val, pal, ...s };
  }
  // local refinement: coordinate ascent over each slot's pool
  for (let round = 0; best && round < 3; round++) {
    for (let k = 0; k < 5; k++) for (const c of pools[k]) {
      const pal = best.pal.slice(); pal[k] = c; const s = score(pal);
      if (s.nrm < 18) continue;
      const val = Math.min(s.cvd, 12) + 0.02 * s.nrm;
      if (val > best.val) best = { val, pal, ...s };
    }
  }
  console.log(mode, surface, "5th =", fname, best ? `${best.pal.join(",")}  worst CVD ${best.cvd.toFixed(1)}  worst normal ${best.nrm.toFixed(1)}` : "none");
}
