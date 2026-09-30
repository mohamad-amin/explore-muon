// Extract the explanatory text of the original Atlas template (read-only) so the readable edition stays in sync:
//   - the VIEWS array (id, label, controls, how)
//   - the Principles list (the `items` array inside `principles()`)
// usage: node extract_text.mjs PATH/TO/atlas_template.html   -> prints JSON {views, principles, warnings}
// The two array literals are located by bracket matching (string-aware) and evaluated in an empty VM context.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const src = readFileSync(process.argv[2], "utf8");
const out = { views: null, principles: null, warnings: [] };

function arrayLiteralAfter(text, marker) {
  const at = text.indexOf(marker);
  if (at < 0) return null;
  const start = text.indexOf("[", at + marker.length - 1);
  if (start < 0) return null;
  let depth = 0, quote = null;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (quote) {
      if (c === "\\") { i++; continue; }
      if (c === quote) quote = null;
      continue;
    }
    if (c === '"' || c === "'" || c === "`") { quote = c; continue; }
    if (c === "[") depth++;
    else if (c === "]") { depth--; if (depth === 0) return text.slice(start, i + 1); }
  }
  return null;
}

function evaluate(literal, what) {
  try {
    return vm.runInNewContext(`(${literal})`, Object.create(null), { timeout: 1000 });
  } catch (e) {
    out.warnings.push(`could not evaluate ${what}: ${e.message}`);
    return null;
  }
}

const viewsLit = arrayLiteralAfter(src, "const VIEWS = [");
if (viewsLit) {
  const views = evaluate(viewsLit, "VIEWS");
  if (Array.isArray(views)) {
    out.views = views.filter((v) => v && typeof v.id === "string")
      .map((v) => ({ id: v.id, label: String(v.label || v.id), controls: Array.isArray(v.controls) ? v.controls : [], how: String(v.how || "") }));
  }
} else out.warnings.push("VIEWS array not found");

const pAt = src.indexOf("principles()");
if (pAt >= 0) {
  const itemsLit = arrayLiteralAfter(src.slice(pAt), "const items = [");
  if (itemsLit) {
    const items = evaluate(itemsLit, "principles");
    if (Array.isArray(items) && items.every((x) => Array.isArray(x) && x.length >= 2)) out.principles = items.map((x) => [String(x[0]), String(x[1])]);
    else out.warnings.push("principles items have an unexpected shape");
  } else out.warnings.push("principles items not found");
} else out.warnings.push("principles() renderer not found");

process.stdout.write(JSON.stringify(out));
