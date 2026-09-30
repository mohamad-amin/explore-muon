"""Build the readable Gauss-Newton Atlas (presentation only; every number comes from the original builder).

usage: build_readable_atlas.py [--out OUT_HTML] [--reuse-original]

1. Runs the original builder, logs/muon_spectra/second_order_audit_20260926/make_atlas.py, as a read-only
   subprocess (`python -B`, so no bytecode is written next to it), with its output in build/original_atlas.html.
   --reuse-original skips this and reuses the last build/original_atlas.html.
2. Extracts the JSON payload from that page's <script id="atlas-data"> element byte for byte and checks that it is
   strict JSON (no NaN/Infinity, which JSON.parse rejects).
3. Extracts the view explanations ("how") and the Principles text from the original atlas_template.html with
   tools/extract_text.mjs (node), so the readable page follows the original's current wording.
4. Writes readable_template.html with __ATLAS_DATA__ (the unmodified payload), __ATLAS_TEXT__ and __ATLAS_META__
   filled in, to gauss_newton_atlas_readable.html by default.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
AUDIT = PROJECT / "logs" / "muon_spectra" / "second_order_audit_20260926"
ORIGINAL_BUILDER = AUDIT / "make_atlas.py"
ORIGINAL_TEMPLATE = AUDIT / "atlas_template.html"
TEMPLATE = HERE / "readable_template.html"
BUILD = HERE / "build"
LIMIT_BYTES = 16 * 1024 * 1024

# Top-level payload keys each view reads (checked at build time; see also tools/check_keys.mjs).
VIEW_KEYS = {
    "principles": [], "curv": ["marg"], "train": ["marg"], "gap": ["gap"], "spec": ["marg"], "gn": ["gn"], "lags": ["lags"],
    "coupling": ["split"], "blocks": ["blockgn", "eoslin", "eos"], "momentum": ["midpoint", "transport"],
    "gap2gn": ["gap2gn"], "edge": ["edge"], "steprule": ["steprule"], "batch": ["batch"],
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_original(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(ORIGINAL_BUILDER), str(out)]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(HERE))


def extract_payload(page: str) -> str:
    m = re.search(r'<script id="atlas-data"[^>]*>(.*?)</script>', page, re.S)
    if not m:
        raise SystemExit("could not find <script id=\"atlas-data\"> in the original page")
    return m.group(1)


def strict_json(raw: str):
    def reject(token):
        raise ValueError(f"payload contains {token}, which JSON.parse rejects")
    return json.loads(raw.replace("<\\/", "</"), parse_constant=reject)


def extract_text() -> dict:
    try:
        res = subprocess.run(["node", str(HERE / "tools" / "extract_text.mjs"), str(ORIGINAL_TEMPLATE)],
                             check=True, capture_output=True, text=True, timeout=60)
        text = json.loads(res.stdout)
    except Exception as e:  # the page falls back to its built-in copies of the texts
        print(f"warning: could not extract texts from the original template ({e}); using built-in copies")
        return {}
    for w in text.get("warnings", []):
        print("warning (texts):", w)
    return {k: text.get(k) for k in ("views", "principles") if text.get(k)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "gauss_newton_atlas_readable.html"))
    ap.add_argument("--reuse-original", action="store_true", help="reuse build/original_atlas.html instead of rebuilding it")
    args = ap.parse_args()

    original_page = BUILD / "original_atlas.html"
    if not args.reuse_original or not original_page.exists():
        run_original(original_page)
    raw = extract_payload(original_page.read_text())
    data = strict_json(raw)
    print(f"payload: {len(raw.encode()) / 1e6:.2f} MB, top-level keys: {', '.join(data)}")
    for view, keys in VIEW_KEYS.items():
        missing = [k for k in keys if not data.get(k)]
        if missing:
            print(f"note: view '{view}' has no data under {missing}; the page shows a 'no data yet' note there")

    text = extract_text()
    if text.get("views"):
        extra = [v["id"] for v in text["views"] if v["id"] not in VIEW_KEYS]
        if extra:
            print(f"note: the original Atlas has views without a readable layout yet: {extra}")
    now = dt.datetime.now(ZoneInfo("America/Chicago"))
    meta = {
        "built": now.strftime("%Y-%m-%d %H:%M %Z"),
        "builder": str(ORIGINAL_BUILDER.relative_to(PROJECT)),
        "builder_sha256": sha256(ORIGINAL_BUILDER.read_bytes()),
        "original_template_sha256": sha256(ORIGINAL_TEMPLATE.read_bytes()),
        "payload_sha256": sha256(raw.encode()),
        "payload_mb": round(len(raw.encode()) / 1e6, 2),
    }
    dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = TEMPLATE.read_text()
    for token in ("__ATLAS_DATA__", "__ATLAS_TEXT__", "__ATLAS_META__"):
        if template.count(token) != 1:
            raise SystemExit(f"template must contain {token} exactly once")
    # the payload is inserted exactly as the original builder serialized it
    page = template.replace("__ATLAS_TEXT__", dump(text)).replace("__ATLAS_META__", dump(meta)).replace("__ATLAS_DATA__", raw)
    out = Path(args.out)
    out.write_text(page)
    size = out.stat().st_size
    print(f"wrote {out}: {size / 1e6:.2f} MB ({'OK' if size < LIMIT_BYTES else 'OVER'} the 16 MB artifact limit)")
    if extract_payload(out.read_text()) != raw:
        raise SystemExit("embedded payload differs from the original payload")
    print("check: embedded payload is byte-identical to the original builder's payload")
    if size >= LIMIT_BYTES:
        raise SystemExit("page is over 16 MB")


if __name__ == "__main__":
    main()
