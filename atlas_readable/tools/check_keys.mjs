// Static check: which top-level payload keys each view's renderer reads, and whether they exist in the payload.
// usage: node tools/check_keys.mjs gauss_newton_atlas_readable.html
import { readFileSync } from "node:fs";

const html = readFileSync(process.argv[2] || "gauss_newton_atlas_readable.html", "utf8");
const data = JSON.parse(html.match(/<script id="atlas-data"[^>]*>([\s\S]*?)<\/script>/)[1].replace(/<\\\//g, "</"));
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).pop();

// split the script into renderer bodies: R.<id> = (...) => { ... } up to the next "R." definition
const starts = [...script.matchAll(/\n  R\.(\w+) = /g)].map((m) => ({ id: m[1], at: m.index }));
let failures = 0;
console.log("view        top-level payload keys read            present?");
starts.forEach((s, i) => {
  const body = script.slice(s.at, i + 1 < starts.length ? starts[i + 1].at : script.indexOf("// ------------------------------------------------------------------ chrome"));
  const keys = [...new Set([...body.matchAll(/\bD\.(\w+)|\bD\["([^"]+)"\]/g)].map((m) => m[1] || m[2]))];
  const missing = keys.filter((k) => !(k in data));
  if (missing.length) failures++;
  console.log(`${s.id.padEnd(11)} ${keys.join(", ").padEnd(38)} ${missing.length ? "MISSING: " + missing.join(", ") : "all present"}`);
});
// second-level keys for the structured sections, as read through local aliases (S = D.steprule, E = D.edge, G = D.gap2gn)
const aliases = { S: "steprule", E: "edge", G: "gap2gn" };
for (const [alias, top] of Object.entries(aliases)) {
  const reads = [...new Set([...script.matchAll(new RegExp(`\\b${alias}\\.(\\w+)\\s*\\|\\|`, "g"))].map((m) => m[1]))];
  const miss = reads.filter((k) => !(data[top] && k in data[top]));
  console.log(`${top}.*: ${reads.join(", ")} -> ${miss.length ? "missing now: " + miss.join(", ") + " (renderer shows a note)" : "all present"}`);
}
const blocks = data.blockgn || {};
console.log(`blockgn.*: blocks, staged -> ${["blocks", "staged"].filter((k) => !(k in blocks)).join(", ") || "all present"}`);
process.exit(failures ? 1 : 0);
