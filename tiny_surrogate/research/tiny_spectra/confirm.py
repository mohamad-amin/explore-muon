"""One-shot, whole-family confirmation; never use a real panel in unit tests.

PLAN: {panel:{path,sha256,kind?}, metrics:{...}, criteria:{...}, runs:[{run_id,
run_dir,config_path,source_root,source_manifest,checkpoint_steps:[0,...,final]}]}.
Paths must be absolute. Configs must enable keep_model_every. The plan lists the
entire checkpoint family produced by that policy. Create the lock BEFORE any
run directory exists. All training runs must finish before ANY score is made.

CLI: lock PLAN OUT_LOCK; score --lock LOCK --run-id ID --out NEW_DIRECTORY;
collect --lock LOCK --scores SCORE_ROOT --out NEW_DIRECTORY. Score prints status
only. It writes sealed score files, which must not be inspected before collect.
Collection releases every result together, without selection or claim testing.
Failed attempts remain intact; a retry needs a new output path and identical
family weights. A completed score cannot be rerun. CPU is synthetic-tests-only.
At the first scoring attempt, the panel is permanently bound to one lock and
complete weight family. Binding and attempt receipts live in panel-local hidden
metadata, so changing or copying the lock filename cannot reopen the panel.
The default panel kind is tiny_shakespeare_char_v1; TinyStories requires explicit
kind=tiny_stories_byte_bpe_v1 and its pinned all-target v2 evaluation manifest.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch
from torch.nn import functional as F

from . import heldout, stories, fineweb, fineweb_panel, model as tiny_model
from research.adamw_spectra import model as production_model


REAL_PANEL_SHA256 = "28fb707e504b3cc8be2aef6784f420cbab406def6e5ffd8a052f8a435694b3af"
STORIES_PANEL_SHA256 = "3dee84c035b4fccc11191efddf83fc4c7b9202a8e4e25d887f46ee2bac44f2bd"
STORIES_TRAINING_SHA256 = "15a4968d198013e014c8666d602fcdb3e1e1a66f01c3a2ed98ea5ebab2660b1d"
# Register only after independent verification of completed Candidate-D data.
# Unset pins reject FineWeb access before its panel loader opens any assets.
FINEWEB_PANEL_SHA256 = None
FINEWEB_TRAINING_SHA256 = None
CHARACTER_KIND = "tiny_shakespeare_char_v1"
STORIES_KIND = "tiny_stories_byte_bpe_v1"
FINEWEB_KIND = "fineweb_byte_bpe_v1"
REAL_PANEL_PATHS = {
    CHARACTER_KIND: Path(__file__).resolve().parents[2] / "data/tiny_shakespeare_confirmation_20260928",
    STORIES_KIND: Path(__file__).resolve().parents[2] / "data/tiny_stories_20260928/sealed_test_v2",
    FINEWEB_KIND: Path(__file__).resolve().parents[2] / "data/fineweb_d_20260928/sealed_test",
}
SOURCE_MODULES = (__file__, heldout.__file__, tiny_model.__file__, production_model.__file__)
TRAINING_SOURCES = {"research/tiny_spectra/" + name + ".py" for name in ("train", "model", "data", "optim")}
TRAINING_SOURCES |= {"research/adamw_spectra/" + name + ".py" for name in ("model", "muon", "data_norm_muon")}
STORIES_SOURCE = "research/tiny_spectra/stories.py"
FINEWEB_SOURCE = "research/tiny_spectra/fineweb.py"


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def absolute(path):
    if not Path(path).is_absolute():
        raise ValueError("All plan paths must be absolute")
    return str(Path(path).resolve())


def verify_sources(root, manifest):
    for relative, expected in manifest.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(Path(root) / path) != expected:
            raise ValueError("Frozen source mismatch: " + relative)


def checkpoint_steps(cfg):
    every = cfg.get("keep_model_every", 0)
    if type(every) is not int or every < 1:
        raise ValueError("Confirmation requires model snapshots")
    steps = math.ceil(cfg["total_tokens"] / cfg["batch_tokens"])
    return sorted({0, 1, steps, *range(every, steps + 1, every)})


def panel_kind(spec):
    kind = spec.get("kind", CHARACTER_KIND)
    if kind not in (CHARACTER_KIND, STORIES_KIND, FINEWEB_KIND):
        raise ValueError("Unsupported confirmation panel kind")
    return kind


def is_stories(panel):
    return panel.manifest.get("corpus_kind") == STORIES_KIND


def is_bpe(panel):
    return panel.manifest.get("corpus_kind") in (STORIES_KIND, FINEWEB_KIND)


def evaluator_source_files(spec):
    kind = panel_kind(spec)
    extra = (fineweb_panel.EVALUATOR_DEPENDENCIES if kind == FINEWEB_KIND else
             (stories.__file__,) if kind == STORIES_KIND else ())
    return tuple(dict.fromkeys(SOURCE_MODULES + extra))


def load_panel(spec):
    kind = panel_kind(spec)
    if kind == FINEWEB_KIND:
        if FINEWEB_PANEL_SHA256 is None or FINEWEB_TRAINING_SHA256 is None:
            raise ValueError("FineWeb confirmation pins are not registered after completed preparation")
        if spec["sha256"] != FINEWEB_PANEL_SHA256:
            raise ValueError("FineWeb confirmation requires its registered panel manifest")
        panel = fineweb_panel.load_panel(spec["path"], expected_manifest_sha256=spec["sha256"])
        if panel.manifest.get("corpus_kind") != FINEWEB_KIND or panel.manifest.get("evaluation_policy_version") != 2:
            raise ValueError("FineWeb requires its own all-target v2 panel kind")
        expected = FINEWEB_PANEL_SHA256
    elif kind == STORIES_KIND:
        panel = stories.load_test_panel(spec["path"], expected_manifest_sha256=spec["sha256"])
        if not is_stories(panel) or panel.manifest.get("evaluation_policy_version") != 2:
            raise ValueError("TinyStories requires its all-target v2 panel kind")
        if any(doc.curve_starts.tolist() != doc.full_starts.tolist()
               or doc.curve_lengths.tolist() != doc.full_lengths.tolist() for doc in panel.documents):
            raise ValueError("TinyStories v2 curves must contain every full target window")
        expected = STORIES_PANEL_SHA256
    else:
        panel = heldout.load_panel(spec["path"], expected_manifest_sha256=spec["sha256"])
        if panel.manifest.get("corpus_kind", CHARACTER_KIND) != CHARACTER_KIND:
            raise ValueError("Character panel kind differs from its manifest")
        expected = REAL_PANEL_SHA256
    if is_bpe(panel) and any(doc.curve_starts.tolist() != doc.full_starts.tolist()
                            or doc.curve_lengths.tolist() != doc.full_lengths.tolist() for doc in panel.documents):
        raise ValueError("BPE confirmation curves must contain every full target window")
    if panel.manifest["role"] not in ("synthetic_contract_test", "sealed_confirmation"):
        raise ValueError("Unsupported confirmation panel role")
    if panel.manifest_sha256 != spec["sha256"]:
        raise ValueError("Panel identity differs from requested manifest")
    if panel.manifest["role"] != "synthetic_contract_test" and spec["sha256"] != expected:
        raise ValueError("Real confirmation requires the pinned external panel")
    if (panel.manifest["role"] != "synthetic_contract_test"
            and Path(panel.path).resolve() != REAL_PANEL_PATHS[kind].resolve()):
        raise ValueError("Real confirmation requires the canonical pinned panel path")
    return panel


def verify_stories_training_data(cfg, panel):
    """Verify frozen train/dev assets and their identity with the sealed tokenizer.

    This reads no extra test data and performs no model operation. Unlike a
    manifest-only checksum, it detects changed token or tokenizer files as well.
    """
    path = Path(cfg["data_path"])
    if cfg.get("corpus_kind") != STORIES_KIND or not path.is_file() or digest(path) != cfg["data_sha256"]:
        raise ValueError("TinyStories training data kind/manifest mismatch")
    if panel.manifest["role"] != "synthetic_contract_test" and cfg["data_sha256"] != STORIES_TRAINING_SHA256:
        raise ValueError("Real TinyStories confirmation requires the pinned training manifest")
    manifest = read(path)
    if (manifest.get("role") != "training_and_development_only"
            or manifest.get("evaluation_policy_version") != 2 or set(manifest.get("splits", {})) != {"train", "val"}):
        raise ValueError("TinyStories training requires the separate all-target v2 data manifest")
    identity_fields = ("corpus_kind", "evaluation_policy_version", "vocabulary", "vocab_size", "seq_len",
                       "eot_id", "tokenizer_sha256", "tokenizer_fit", "membership_sha256", "prefix_hashes", "revision")
    for name in identity_fields:
        if name not in manifest or manifest[name] != panel.manifest.get(name):
            raise ValueError("Training/panel data identity differs: " + name)
    assets = [manifest[name] for name in ("tokenizer", "default_permutation", "dev_starts", "dev_lengths", "dev_document_indices")]
    assets += [split[name] for split in manifest["splits"].values() for name in ("tokens", "documents")]
    for spec in assets:
        relative = Path(spec["path"])
        asset_path = (path.parent / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not asset_path.is_relative_to(path.parent.resolve())
                or asset_path.stat().st_size != spec["bytes"] or digest(asset_path) != spec["sha256"]):
            raise ValueError("TinyStories training artifact integrity failure")
    if manifest["tokenizer"]["sha256"] != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer artifact and data identity disagree")
    identity = dict(manifest_sha256=cfg["data_sha256"], tokenizer_sha256=manifest["tokenizer_sha256"],
                    membership_sha256=manifest["membership_sha256"], evaluation_policy_version=2,
                    corpus_kind=STORIES_KIND)
    return manifest, identity


def verify_bpe_training_data(cfg, panel):
    if is_stories(panel):
        return verify_stories_training_data(cfg, panel)
    if (panel.manifest.get("corpus_kind") != FINEWEB_KIND or FINEWEB_TRAINING_SHA256 is None
            or cfg.get("data_sha256") != FINEWEB_TRAINING_SHA256):
        raise ValueError("FineWeb confirmation requires its registered training manifest")
    return fineweb_panel.verify_training_data(cfg, panel)


def create_lock(plan_path, out):
    if Path(out).exists():
        raise FileExistsError("Lock already exists")
    plan = read(plan_path)
    if not plan.get("metrics") or not plan.get("criteria") or not plan.get("runs"):
        raise ValueError("Lock requires complete metrics, criteria and run family")
    plan["panel"]["path"] = absolute(plan["panel"]["path"])
    panel = load_panel(plan["panel"])
    if is_bpe(panel) and plan["metrics"].get("primary") != "token_weighted_nll":
        raise ValueError("BPE primary metric must be token_weighted_nll")
    seen, directories = set(), set()
    for run in plan["runs"]:
        name = run["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", "..") or name in seen:
            raise ValueError("Unique safe run IDs required")
        seen.add(name)
        for key in ("run_dir", "config_path", "source_root", "source_manifest"):
            run[key] = absolute(run[key])
        if Path(run["run_dir"]).exists() or run["run_dir"] in directories:
            raise ValueError("Lock must precede every unique training run directory")
        directories.add(run["run_dir"])
        cfg = read(run["config_path"])
        if cfg.get("run_id") != name or cfg["seq_len"] != panel.seq_len:
            raise ValueError("Run ID or context differs from panel")
        if run["checkpoint_steps"] != checkpoint_steps(cfg):
            raise ValueError("Plan must include the entire saved checkpoint family")
        run.update(config=cfg, config_sha256=digest(run["config_path"]),
                   sources=read(run["source_manifest"]), manifest_sha256=digest(run["source_manifest"]))
        required_sources = TRAINING_SOURCES | ({STORIES_SOURCE} if is_bpe(panel) else set())
        if panel.manifest.get("corpus_kind") == FINEWEB_KIND:
            required_sources |= {FINEWEB_SOURCE}
        if not required_sources <= set(run["sources"]):
            raise ValueError("Source manifest does not cover the training implementation")
        verify_sources(run["source_root"], run["sources"])
        modules = (tiny_model, production_model) + ((stories,) if is_bpe(panel) else ())
        if panel.manifest.get("corpus_kind") == FINEWEB_KIND:
            modules += (fineweb,)
        for module in modules:
            relative = str(Path(module.__file__).resolve().relative_to(Path(__file__).resolve().parents[2]))
            if run["sources"].get(relative) != digest(module.__file__):
                raise ValueError("Evaluation model source differs from training source")
        if digest(cfg["data_path"]) != cfg["data_sha256"]:
            raise ValueError("Training data differs from config")
        if is_bpe(panel):
            _, run["data_identity"] = verify_bpe_training_data(cfg, panel)
        elif cfg.get("corpus_kind") in (STORIES_KIND, FINEWEB_KIND):
            raise ValueError("BPE training cannot use a character confirmation panel")
    plan.update(evaluator_sources={absolute(p): digest(p) for p in evaluator_source_files(plan["panel"])},
                torch_version=torch.__version__, plan_sha256=digest(plan_path))
    payload = dict(schema_version=1, plan=plan)
    write_new(out, dict(payload=payload, sha256=hashlib.sha256(canonical(payload)).hexdigest()))
    return read(out)


def load_lock(path):
    lock = read(path)
    if hashlib.sha256(canonical(lock["payload"])).hexdigest() != lock["sha256"]:
        raise ValueError("Lock integrity failure")
    plan = lock["payload"]["plan"]
    if lock["payload"]["schema_version"] != 1 or plan["torch_version"] != torch.__version__:
        raise ValueError("Evaluation schema/environment changed")
    for source, expected in plan["evaluator_sources"].items():
        if digest(source) != expected:
            raise ValueError("Evaluator source changed after lock")
    if {absolute(p) for p in evaluator_source_files(plan["panel"])} != set(plan["evaluator_sources"]):
        raise ValueError("Evaluator loaded from a different source tree")
    return lock, load_panel(plan["panel"])


def complete_family(lock, panel):
    """Verify all completed runs and inventory all required weights before scoring."""
    family = {}
    for run in lock["payload"]["plan"]["runs"]:
        root, cfg = Path(run["run_dir"]), run["config"]
        if not (root / "summary.json").exists() or not (root / "status.json").exists():
            raise ValueError("Whole training family must complete before scoring")
        summary, status, metadata = (read(root / name) for name in ("summary.json", "status.json", "metadata.json"))
        if any(value.get("status") != "complete" for value in (summary, status)):
            raise ValueError("Whole training family must complete before scoring")
        if (digest(run["config_path"]) != run["config_sha256"] or read(root / "config.json") != cfg
                or digest(run["source_manifest"]) != run["manifest_sha256"]):
            raise ValueError("Locked config/source manifest changed")
        verify_sources(run["source_root"], run["sources"])
        if digest(cfg["data_path"]) != cfg["data_sha256"]:
            raise ValueError("Training data changed")
        if is_bpe(panel):
            data_manifest, identity = verify_bpe_training_data(cfg, panel)
            if (identity != run.get("data_identity") or metadata.get("data_manifest_sha256") != cfg["data_sha256"]
                    or metadata.get("corpus") != data_manifest):
                raise ValueError("Training metadata/data identity differs from locked BPE data")
        final_step = run["checkpoint_steps"][-1]
        if summary["steps"] != final_step or summary["tokens"] != cfg["total_tokens"]:
            raise ValueError("Incomplete locked horizon")
        expected_source = str(Path(run["source_root"]) / "research/tiny_spectra/train.py")
        if metadata["source_file"] != expected_source or metadata["corpus"]["vocabulary"] != list(panel.vocabulary):
            raise ValueError("Training source or vocabulary mismatch")
        if (any(metadata["model_config"][key] != cfg[key] for key in ("n_layer", "n_embd", "n_head", "seq_len"))
                or metadata["model_config"]["vocab_size"] != len(panel.vocabulary)):
            raise ValueError("Training architecture differs from locked config")
        if panel.manifest["role"] != "synthetic_contract_test":
            if not 0 < metadata["hardware"]["total_memory_bytes"] < 45 * 1024**3:
                raise ValueError("Training hardware violates the VRAM restriction")
        files = {str(step): digest(root / "models" / f"step{step:06d}.pt") for step in run["checkpoint_steps"]}
        template = None
        for step in run["checkpoint_steps"]:
            template = load_snapshot(run, step, files[str(step)], torch.device("cpu"), model=template)
        family[run["run_id"]] = dict(models=files, metadata=digest(root / "metadata.json"),
                                    summary=digest(root / "summary.json"), status=digest(root / "status.json"))
    return family


def load_snapshot(run, step, expected, device, model=None):
    path = Path(run["run_dir"]) / "models" / f"step{step:06d}.pt"
    if digest(path) != expected:
        raise ValueError("Snapshot changed during evaluation")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = run["config"]
    if (saved["format"] != "tiny_spectra_model_only_v1" or saved["step"] != step
            or saved["tokens"] != min(step * cfg["batch_tokens"], cfg["total_tokens"])):
        raise ValueError("Snapshot step/horizon mismatch")
    if saved["model_config"] != read(Path(run["run_dir"]) / "metadata.json")["model_config"]:
        raise ValueError("Snapshot architecture mismatch")
    if model is None:
        with torch.random.fork_rng(devices=[]):
            model = tiny_model.GPT(tiny_model.ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"], strict=True)
    hashed = hashlib.sha256()
    for name, parameter in model.named_parameters():
        hashed.update(name.encode())
        hashed.update(parameter.detach().contiguous().numpy().tobytes())
        if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
            raise ValueError("Snapshot requires finite FP32 weights")
    if hashed.hexdigest() != saved["parameter_sha256"]:
        raise ValueError("Snapshot parameter hash mismatch")
    return model.to(device).eval()


@torch.no_grad()
def evaluate_document(model, document, starts, seq_len, device, microbatch=32, *, lengths=None, eot_id=None):
    tokens = torch.tensor(document.token_ids.copy(), dtype=torch.long)
    values = []
    for begin in range(0, len(starts), microbatch):
        if lengths is not None:
            if len(lengths) != len(starts) or eot_id is None:
                raise ValueError("Masked evaluation requires aligned lengths and explicit EOT")
            x, y = stories.padded_evaluation_batch(tokens, starts[begin:begin + microbatch].copy(),
                                                   lengths[begin:begin + microbatch].copy(), seq_len, eot_id, device)
            counts = (y != -100).sum(1)
        else:
            indices = torch.tensor(starts[begin:begin + microbatch].copy())[:, None] + torch.arange(seq_len + 1)
            windows = tokens[indices].to(device)
            if (windows < 0).any():
                raise ValueError("OOV reached scoring; no hidden dropping allowed")
            x, y = windows[:, :-1], windows[:, 1:]
            counts = torch.full((len(windows),), seq_len, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.reshape(-1), reduction="none", ignore_index=-100)
        if not torch.isfinite(loss).all():
            raise ValueError("Nonfinite confirmation loss")
        values.extend((loss.double().view(-1, seq_len).sum(1) / counts).cpu().tolist())
    return values


def aggregate(per_document, panel):
    if is_bpe(panel):
        weights = {doc.slug: doc.manifest["counts"]["full_target_tokens"] for doc in panel.documents}
        if set(per_document) != set(weights) or not weights or min(weights.values()) <= 0:
            raise ValueError("Incomplete document scores or invalid token weights")
        return dict(token_weighted_nll=sum(per_document[name] * weight for name, weight in weights.items()) / sum(weights.values()),
                    macro_document_nll=sum(per_document.values()) / len(weights),
                    full_target_tokens=sum(weights.values()), scored_documents=len(weights), target_unit="BPE_token")
    weights = {doc.slug: doc.manifest["counts"]["full_target_characters"] for doc in panel.documents}
    return dict(character_weighted_nll=sum(per_document[name] * weight for name, weight in weights.items()) / sum(weights.values()),
                macro_nll=sum(per_document.values()) / len(weights), full_target_characters=sum(weights.values()))


def evaluate_bank(model, document, bank, panel, device):
    starts = document.curve_starts if bank == "curve" else document.full_starts
    if is_bpe(panel):
        lengths = document.curve_lengths if bank == "curve" else document.full_lengths
        if lengths is None or len(lengths) != len(starts):
            raise ValueError("Missing BPE target lengths")
        values = evaluate_document(model, document, starts, panel.seq_len, device,
                                   lengths=lengths, eot_id=panel.manifest["eot_id"])
        counts = lengths.tolist()
    else:
        values = evaluate_document(model, document, starts, panel.seq_len, device)
        counts = [panel.seq_len] * len(starts)
    if not counts or min(counts) <= 0:
        raise ValueError("Empty or invalid scoring document")
    mean = sum(value * count for value, count in zip(values, counts)) / sum(counts)
    row = dict(starts=starts.tolist(), nll=values)
    if is_bpe(panel):
        row.update(lengths=counts, target_tokens=sum(counts))
    return mean, row


@torch.no_grad()
def evaluate_bpe_panel(model, panel, device, microbatch=32):
    """Pool BPE windows across documents; padding never crosses a document.

    A token concatenation is only storage: every start/length pair is checked
    against its original document before batching. Reduction preserves each
    document's actual target count and every individual window loss.
    """
    if not is_bpe(panel) or microbatch < 1:
        raise ValueError("Positive microbatch and BPE panel required")
    pieces, starts, lengths, owners, rows = [], [], [], [], {}
    offset = 0
    for document in panel.documents:
        source = torch.tensor(document.token_ids.copy(), dtype=torch.long)
        local_starts = document.full_starts.tolist()
        local_lengths = document.full_lengths.tolist()
        if (document.slug in rows or len(local_starts) != len(local_lengths) or not local_starts
                or any(length < 1 or length > panel.seq_len or start < 0 or start + length >= len(source)
                       for start, length in zip(local_starts, local_lengths))
                or sum(local_lengths) != document.manifest["counts"]["full_target_tokens"]):
            raise ValueError("Invalid document boundaries or target counts in pooled scoring")
        pieces.append(source)
        starts.extend(start + offset for start in local_starts)
        lengths.extend(local_lengths)
        owners.extend([document.slug] * len(local_starts))
        rows[document.slug] = dict(starts=local_starts, lengths=local_lengths,
                                  target_tokens=sum(local_lengths), nll=[])
        offset += len(source)
    if not pieces:
        raise ValueError("Empty BPE scoring panel")
    tokens = torch.cat(pieces)
    if (tokens < 0).any() or (tokens >= len(panel.vocabulary)).any():
        raise ValueError("Invalid source token reached pooled BPE scoring")
    for begin in range(0, len(starts), microbatch):
        end = begin + microbatch
        x, y = stories.padded_evaluation_batch(tokens, starts[begin:end], lengths[begin:end],
                                               panel.seq_len, panel.manifest["eot_id"], device)
        with torch.autocast(device_type=device.type, enabled=False):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.reshape(-1), reduction="none", ignore_index=-100)
        if not torch.isfinite(loss).all():
            raise ValueError("Nonfinite confirmation loss")
        means = (loss.double().view(-1, panel.seq_len).sum(1) / (y != -100).sum(1)).cpu().tolist()
        for owner, value in zip(owners[begin:end], means):
            rows[owner]["nll"].append(value)
    per_document = {name: sum(value * count for value, count in zip(row["nll"], row["lengths"])) / row["target_tokens"]
                    for name, row in rows.items()}
    return per_document, rows


def evaluate_stories_panel(model, panel, device, microbatch=32):
    """Compatibility entry point for the existing TinyStories contracts."""
    if not is_stories(panel):
        raise ValueError("TinyStories panel required")
    return evaluate_bpe_panel(model, panel, device, microbatch)


def binding_identity(lock, panel, family):
    return dict(schema_version=1, panel_manifest_sha256=panel.manifest_sha256,
                lock_sha256=lock["sha256"],
                family_sha256=hashlib.sha256(canonical(family)).hexdigest())


def valid_binding_identity(value):
    fields = {"schema_version", "panel_manifest_sha256", "lock_sha256", "family_sha256"}
    return (isinstance(value, dict) and set(value) == fields
            and type(value["schema_version"]) is int and value["schema_version"] == 1
            and all(isinstance(value[key], str) and re.fullmatch(r"[0-9a-f]{64}", value[key])
                    for key in fields - {"schema_version"}))


def bind_panel(panel, identity, create=True):
    """Publish a complete one-family receipt atomically, without replacement.

    A failed score never releases the binding. Requiring the canonical real
    panel path also prevents accidental reuse of bare copied panel assets.
    This is an audit safeguard, not protection against deliberate file deletion.
    """
    if not valid_binding_identity(identity) or identity["panel_manifest_sha256"] != panel.manifest_sha256:
        raise ValueError("Invalid confirmation panel binding identity")
    metadata = Path(panel.path).resolve() / ".confirmation"
    receipt = metadata / "family.json"
    if create:
        metadata.mkdir(exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=metadata, prefix=".family.", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(canonical(identity))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, receipt)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink()
    try:
        saved = read(receipt)
    except (OSError, ValueError) as error:
        raise ValueError("Missing or malformed confirmation panel binding") from error
    if not valid_binding_identity(saved):
        raise ValueError("Malformed confirmation panel binding")
    if saved != identity:
        raise ValueError("Only identical lock/weights may use this bound confirmation panel")
    return metadata


@contextmanager
def scoring_attempt(lock_path, lock, run_id, family, out, *, panel):
    identity = binding_identity(lock, panel, family)
    registry = bind_panel(panel, identity) / "attempts"
    registry.mkdir(exist_ok=True)
    guard = registry / (run_id + ".active")
    with guard.open("x"):
        pass
    try:
        for receipt in registry.glob(run_id + ".attempt.*.json"):
            previous = read(receipt)
            if previous["identity"] != identity or read(Path(previous["out"]) / "status.json").get("status") != "failed":
                raise ValueError("Only failed attempts with identical lock/weights may retry")
        out.mkdir(parents=True, exist_ok=False)
        write_new(out / "status.json", dict(status="running", **identity, run_id=run_id))
        receipt = registry / (run_id + ".attempt." + hashlib.sha256(str(out).encode()).hexdigest() + ".json")
        write_new(receipt, dict(identity=identity, out=str(out)))
        try:
            yield identity
        except Exception as error:
            write_new(out / "failure.json", dict(error=repr(error), **identity))
            (out / "status.json").write_bytes(canonical(dict(status="failed", **identity, run_id=run_id)))
            raise
    finally:
        guard.unlink()


def score(lock_path, run_id, out, device="cuda"):
    lock, panel = load_lock(lock_path)
    family = complete_family(lock, panel)
    run = next((r for r in lock["payload"]["plan"]["runs"] if r["run_id"] == run_id), None)
    if run is None:
        raise ValueError("Run is outside the locked family")
    device = torch.device(device)
    if device.type == "cpu" and panel.manifest["role"] != "synthetic_contract_test":
        raise ValueError("CPU scoring is restricted to synthetic tests")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Unsupported evaluation device")
    hardware = {"name": "CPU synthetic qualification", "total_memory_bytes": None}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory >= 45 * 1024**3:
            raise ValueError("Evaluation GPU must be strictly below nominal 48GB")
        hardware = dict(name=properties.name, total_memory_bytes=properties.total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Scoring requires a new output directory")
    with scoring_attempt(lock_path, lock, run_id, family, out, panel=panel) as identity:
        curves, final_windows = [], {}
        document_key = "per_document" if is_bpe(panel) else "per_play"
        for step in run["checkpoint_steps"]:
            model = load_snapshot(run, step, family[run_id]["models"][str(step)], device)
            if is_stories(panel):
                per_document, pooled_windows = evaluate_stories_panel(model, panel, device)
            elif is_bpe(panel):
                per_document, pooled_windows = evaluate_bpe_panel(model, panel, device)
            else:
                per_document = {doc.slug: evaluate_bank(model, doc, "curve", panel, device)[0] for doc in panel.documents}
            curves.append(dict(step=step, tokens=min(step * run["config"]["batch_tokens"], run["config"]["total_tokens"]),
                               **{document_key: per_document}, **aggregate(per_document, panel)))
            if step == run["checkpoint_steps"][-1]:
                final_windows = (pooled_windows if is_bpe(panel) else
                                 {doc.slug: evaluate_bank(model, doc, "full", panel, device)[1] for doc in panel.documents})
            del model
        final = {name: (sum(value * count for value, count in zip(row["nll"], row["lengths"])) / row["target_tokens"]
                        if is_bpe(panel) else sum(row["nll"]) / len(row["nll"]))
                 for name, row in final_windows.items()}
        denominators = (dict(full_windows=panel.manifest["totals"]["full_windows"],
                             curve_windows=panel.manifest["totals"]["curve_windows"],
                             full_target_tokens=panel.manifest["totals"]["full_target_tokens"],
                             curve_target_tokens=panel.manifest["totals"]["curve_target_tokens"],
                             scored_documents=len(panel.documents), target_unit="BPE_token")
                        if is_bpe(panel) else panel.manifest["totals"])
        result = dict(**identity, run_id=run_id, panel_sha256=panel.manifest_sha256, family=family,
                      precision="FP32; TF32 disabled", device=str(device), hardware=hardware, curves=curves,
                      final=dict(**{document_key: final}, windows=final_windows, **aggregate(final, panel)),
                      panel_denominators=denominators)
        if is_bpe(panel):
            result.update(corpus_kind=panel.manifest["corpus_kind"], target_unit="BPE_token", evaluation_policy_version=2,
                          data_identity=run["data_identity"])
        write_new(out / "sealed_scores.json", result)
        (out / "status.json").write_bytes(canonical(dict(status="complete", **identity, run_id=run_id,
                                                       score_sha256=digest(out / "sealed_scores.json"))))
    return {"status": "scored_and_sealed", "run_id": run_id}


def collect(lock_path, scores, out):
    lock, panel = load_lock(lock_path)
    family = complete_family(lock, panel)
    bind_panel(panel, binding_identity(lock, panel, family), create=False)
    expected = {r["run_id"] for r in lock["payload"]["plan"]["runs"]}
    completed = {}
    for path in Path(scores).rglob("sealed_scores.json"):
        status = read(path.parent / "status.json")
        if status.get("status") != "complete" or status.get("lock_sha256") != lock["sha256"]:
            continue
        row = read(path)
        name = row["run_id"]
        if (name in completed or name not in expected or row["family"] != family
                or row["panel_sha256"] != panel.manifest_sha256 or digest(path) != status["score_sha256"]):
            raise ValueError("Duplicate, changed, or foreign confirmation scores")
        completed[name] = row
    if set(completed) != expected:
        raise ValueError("Every locked run must be scored before collection")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "all_results.json", dict(lock=lock, results=[completed[name] for name in sorted(completed)]))
    return {"status": "complete_family_released", "runs": len(completed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    command = commands.add_parser("lock")
    command.add_argument("plan")
    command.add_argument("out")
    for operation in ("score", "collect"):
        command = commands.add_parser(operation)
        command.add_argument("--lock", required=True)
        command.add_argument("--out", required=True)
        if operation == "score":
            command.add_argument("--run-id", required=True)
            command.add_argument("--device", default="cuda")
        else:
            command.add_argument("--scores", required=True)
    args = parser.parse_args()
    if args.operation == "lock":
        create_lock(args.plan, args.out)
        result = {"status": "locked"}
    elif args.operation == "score":
        result = score(args.lock, args.run_id, args.out, args.device)
    else:
        result = collect(args.lock, args.scores, args.out)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
