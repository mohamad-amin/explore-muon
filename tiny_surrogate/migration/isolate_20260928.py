"""One-time, provenance-preserving relocation requested by the user."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

STUDY = Path(__file__).resolve().parents[1]
PARENT = STUDY.parent
RECORD = STUDY / "migration"
RELATIVE_ROOTS = (
    "research/tiny_spectra", "logs/tiny_spectra", "data/tiny_shakespeare",
    "data/tiny_shakespeare_holdout_audit", "data/tiny_shakespeare_confirmation_20260928",
    "data/tiny_stories_20260928",
)
OWN_TITLES = (
    "### Tiny surrogate qualification:", "**Tiny qualification and screen launch",
    "**Tiny Shakespeare first screen complete", "**Tiny surrogate next decision:",
    "**Tiny batch sentinels complete", "**Independent tiny-surrogate review",
    "**Large-batch sweep reading and bounded fairness/closure check",
    "**Execution qualification on another eligible GPU",
    "**User correction: prevent evaluation-selection overfitting",
    "**RTX2080Ti execution qualification passed", "**Character candidate verdict",
    "**Fresh-text subword candidate C:", "**Candidate C qualified and screen submitted",
    "**Prospective reference-premise clarification before candidate C dynamics",
    "**Candidate C endpoint-rate rule fixed before dynamics",
    "**Candidate C initial screen complete; dynamics gate closed",
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def inventory(root):
    result = {}
    for path in [root, *root.rglob("*")]:
        st = path.lstat()
        key = str(path.relative_to(root))
        result[key] = dict(device=st.st_dev, inode=st.st_ino, size=st.st_size,
                           mtime_ns=st.st_mtime_ns, mode=st.st_mode,
                           target=os.readlink(path) if path.is_symlink() else None)
    return result


def isolate_notes():
    protocol = PARENT / "research/adamw_spectra/MUON_CASE.md"
    original_bytes = protocol.read_bytes()
    original = original_bytes.decode()
    lines = original.splitlines(keepends=True)
    begin = next(i for i, line in enumerate(lines) if line.startswith(OWN_TITLES[0]))
    starts = []
    for i, line in enumerate(lines):
        if i < begin:
            continue
        title = line.split("**", 2)[1] if line.startswith("**") else line
        if line.startswith("### ") or (line.startswith("**") and
                ("2026-09-28" in title or line.startswith(OWN_TITLES[14]))):
            starts.append(i)
    starts.append(len(lines))
    exported, entries = [], []
    retained = ["".join(lines[:begin])]
    pointer = ("### Tiny-surrogate study relocated (2026-09-28)\n\n"
               "At the user's request, this branch's code, data, results and decision history now live in "
               "`tiny_surrogate/`. See [its protocol](../../tiny_surrogate/PROTOCOL.md) and "
               "[current state](../../tiny_surrogate/RESEARCH_STATE.md). The original shared journal "
               "is retained in that directory's migration snapshot.\n\n")
    inserted = False
    for left, right in zip(starts, starts[1:]):
        block = "".join(lines[left:right])
        if lines[left].startswith(OWN_TITLES):
            exported.append(block)
            entries.append(dict(heading=lines[left].strip(), start_line=left+1, end_line=right,
                                sha256=sha(block.encode())))
            if not inserted:
                retained.append(pointer)
                inserted = True
        else:
            retained.append(block)
    if len(entries) != len(OWN_TITLES):
        raise RuntimeError(f"Unexpected owned-entry count {len(entries)}")
    (RECORD / "shared_protocol_before.md").write_bytes(original_bytes)
    (STUDY / "PROTOCOL.md").write_text(
        "# Small-surrogate study: protocol and decision history\n\n"
        "The entries below were moved verbatim from the shared project journal on 2026-09-28, "
        "at the user's request. Historical paths refer to the former project root; compatibility "
        "links preserve recorded artifact paths. New work is confined to this directory. "
        "The original reference study remains [in the parent notebook](../research/adamw_spectra/MUON_CASE.md).\n\n"
        + "".join(exported))
    updated = "".join(retained)
    if protocol.read_bytes() != original_bytes:
        raise RuntimeError("Shared protocol changed concurrently; do not overwrite it")
    protocol.write_text(updated)
    state = PARENT / "RESEARCH_STATE.md"
    before_state = state.read_bytes()
    text = before_state.decode()
    left = text.index("**User-directed tiny-surrogate branch (2026-09-28): active, not qualified.**")
    right = text.index("\n\n**Principles so far", left)
    (RECORD / "shared_state_before.md").write_bytes(before_state)
    (STUDY / "RESEARCH_STATE.md").write_text("# Small-surrogate research state\n\n" +
        text[left:right].replace("research/adamw_spectra/MUON_CASE.md", "PROTOCOL.md") + "\n")
    state_pointer = ("**Tiny-surrogate study:** isolated at the user's request in "
                     "[tiny_surrogate/](tiny_surrogate/README.md). Its live state and all new "
                     "decisions are maintained [there](tiny_surrogate/RESEARCH_STATE.md).")
    if state.read_bytes() != before_state:
        raise RuntimeError("Shared state changed concurrently; do not overwrite it")
    state.write_text(text[:left] + state_pointer + text[right:])
    return dict(entries=entries, parent_protocol_before_sha256=sha(original_bytes),
                parent_protocol_after_sha256=sha(protocol.read_bytes()),
                parent_state_before_sha256=sha(before_state),
                parent_state_after_sha256=sha(state.read_bytes()))


def main():
    if (RECORD / "relocation.json").exists():
        raise RuntimeError("Relocation already recorded; inspect, do not repeat")
    ids = {json.loads(p.read_text())["job_id"].split(";")[0]
           for p in (PARENT / "logs/tiny_spectra").glob("*/submission.json")}
    live = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%T"],
                          check=True, capture_output=True, text=True).stdout
    if any(row.split("_", 1)[0].split("|", 1)[0] in ids for row in live.splitlines()):
        raise RuntimeError("An owned Slurm job is still live; defer relocation")
    report = dict(started_unix=time.time(), old_root=str(PARENT), new_root=str(STUDY),
                  owned_job_ids=sorted(ids), live_scheduler_snapshot=live, moved=[], compatibility_links=[])
    before = {relative: inventory(PARENT / relative) for relative in RELATIVE_ROOTS}
    dump(RECORD / "inventory_before.json", before)
    report["notes"] = isolate_notes()
    for relative in RELATIVE_ROOTS:
        old, new = PARENT / relative, STUDY / relative
        if old.is_symlink() or new.exists():
            raise RuntimeError("Unexpected pre-existing relocation path: " + relative)
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        old.symlink_to(new, target_is_directory=True)
        after = inventory(new)
        if after != before[relative]:
            raise RuntimeError("Relocated inode/size/mtime inventory differs: " + relative)
        report["moved"].append(dict(relative=relative, entries=len(after), inode_inventory_unchanged=True))
        report["compatibility_links"].append(dict(path=str(old), target=str(new)))
    # Own immutable numerical dependencies; no parent production trainer is used.
    dependency_root = STUDY / "research/adamw_spectra"
    dependency_root.mkdir(parents=True, exist_ok=False)
    reference = STUDY / "logs/tiny_spectra/stories_c_edges_20260928/frozen/research/adamw_spectra"
    dependencies = {}
    for name in ("__init__.py", "model.py", "muon.py", "data_norm_muon.py"):
        shutil.copy2(reference / name, dependency_root / name)
        dependencies[name] = dict(source=str(reference / name), sha256=sha((dependency_root / name).read_bytes()))
    (dependency_root / "train.py").write_text(
        '"""Compatibility helper for qualified kernel-reference tests; no training pipeline."""\n'
        'import contextlib\nimport torch\n\n'
        'def amp_context(device, config):\n'
        '    if device.type == "cuda" and config["precision"] == "bf16":\n'
        '        return torch.autocast("cuda", dtype=torch.bfloat16)\n'
        '    return contextlib.nullcontext()\n')
    dump(dependency_root / "PROVENANCE.json", dependencies)
    (STUDY / ".venv").symlink_to(PARENT / ".venv", target_is_directory=True)
    shutil.copy2(PARENT / "RESEARCH_GUIDE.md", RECORD / "parent_research_guide.md")
    scratch = STUDY / "scratch/earlier_fixtures"
    scratch.mkdir(parents=True)
    report["earlier_fixtures"] = []
    for name in ("tiny-provenance-r39d_6x1", "tiny-dynamics-report-yz9ns7f2", "tiny-metric-policy-0oznqso8",
                 "tiny-dynamics-bounds-uk0eufwq", "tiny-selection-policy-baseline.json", "tiny-analyzer-contract-hwoak_fq"):
        old, new = Path("/tmp") / name, scratch / name
        if old.exists() and not old.is_symlink():
            shutil.move(str(old), str(new))
            old.symlink_to(new, target_is_directory=new.is_dir())
            report["earlier_fixtures"].append(dict(path=str(old), target=str(new)))
    report["finished_unix"] = time.time()
    dump(RECORD / "relocation.json", report)
    print(json.dumps(dict(moved_roots=len(report["moved"]), protocol_entries=len(report["notes"]["entries"]),
                          old_paths_preserved=True, record=str(RECORD / "relocation.json")), indent=2))


if __name__ == "__main__":
    main()
