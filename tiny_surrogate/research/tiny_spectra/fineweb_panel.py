"""FineWeb confirmation provenance and panel loading; no inference or preparation.

Only explicit callers open a panel. Training identity checks use the prepared
train/development assets and preparation receipts, never a tokenizer dependency.
"""
from contextlib import closing
import json
from pathlib import Path

from . import data, fineweb, stories


KIND = fineweb.KIND
EVALUATOR_DEPENDENCIES = (__file__, fineweb.__file__, stories.__file__, data.__file__)
IDENTITY_FIELDS = (
    "corpus_kind", "evaluation_policy_version", "revision", "source_repo",
    "vocabulary", "vocab_size", "seq_len", "eot_id", "tokenizer_sha256",
    "tokenizer_fit", "roles_frozen_sha256", "preparation", "target_padding_id",
    "input_padding_id", "evaluation_boundary_rule",
)


def read(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError("Missing or malformed FineWeb preparation artifact: " + str(path)) from error


def asset(root, spec):
    root = Path(root).resolve()
    relative = Path(spec["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError("FineWeb prepared asset escaped its data directory")
    if path.stat().st_size != spec["bytes"] or fineweb.file_sha256(path) != spec["sha256"]:
        raise ValueError("FineWeb prepared asset integrity failure: " + str(path))
    return path


def verify_preparation(root, training, panel_manifest, *, training_sha256, panel_sha256):
    """Bind published tokens to a completed, unchanged role/decoder preparation."""
    root = Path(root).resolve()
    if (root / "PREPARATION_FAILED.json").exists():
        raise ValueError("FineWeb preparation has a retained failure")
    complete = read(root / "PREPARATION_COMPLETE.json")
    prepared = read(root / "MANIFESTS_PREPARED.json")
    if (complete != prepared or complete.get("no_model_inference") is not True
            or complete.get("training_manifest_sha256") != training_sha256
            or complete.get("test_manifest_sha256") != panel_sha256
            or complete.get("fresh_target_capacity") != training.get("fresh_target_capacity")):
        raise ValueError("FineWeb preparation completion does not bind these manifests")
    if (training.get("corpus_kind") != KIND or panel_manifest.get("corpus_kind") != KIND
            or training.get("role") != "training_and_development_only"
            or set(training.get("splits", {})) != {"train", "val"}
            or panel_manifest.get("role") not in ("sealed_confirmation", "synthetic_contract_test")
            or set(panel_manifest.get("splits", {})) != {"test"}
            or training.get("evaluation_policy_version") != 2):
        raise ValueError("FineWeb preparation has incompatible data roles or evaluation policy")
    for field in IDENTITY_FIELDS:
        if field not in training or training[field] != panel_manifest.get(field):
            raise ValueError("FineWeb training/panel identity differs: " + field)
    if (training["revision"] != fineweb.SOURCE_REVISION
            or training["source_repo"] != fineweb.SOURCE_REPO
            or training["target_padding_id"] != -100
            or training["input_padding_id"] != training["eot_id"]
            or len(training["vocabulary"]) != training["vocab_size"]):
        raise ValueError("FineWeb source, vocabulary, or padding identity differs")
    roles = read(root / "ROLES_FROZEN.json")
    roles_sha = fineweb.file_sha256(root / "ROLES_FROZEN.json")
    fit = training["tokenizer_fit"]
    if (roles.get("corpus_kind") != KIND or roles.get("roles_frozen_before_tokenizer") is not True
            or roles.get("no_model_scores") is not True
            or set(roles.get("roles", {})) != {"train", "dev", "test"}
            or roles_sha != training["roles_frozen_sha256"]
            or fit.get("roles_frozen_sha256") != roles_sha or fit.get("train_only") is not True
            or fit.get("accepted_training_sha256") != roles["roles"]["train"]["sha256"]
            or fit.get("training_document_count") != roles["counts"]["train"]
            or fit.get("vocab_size") != training["vocab_size"]
            or fit.get("eot_id") != training["eot_id"]
            or read(root / "tokenizer_fit.json") != fit):
        raise ValueError("FineWeb frozen roles or train-only tokenizer identity differs")
    if (complete.get("counts") != roles["counts"]
            or roles["counts"] != dict(train=training["splits"]["train"]["document_count"],
                                       dev=training["splits"]["val"]["document_count"],
                                       test=panel_manifest["splits"]["test"]["document_count"])):
        raise ValueError("FineWeb prepared document-role counts differ")
    # Recheck the exact role inputs and membership that qualified before encoding.
    fineweb.verify_frozen_roles(root, roles)
    for key in ("membership", "overlap_exclusions"):
        asset(root, roles[key])
    preparation = training["preparation"]
    if preparation.get("no_model_inference") is not True:
        raise ValueError("FineWeb preparation provenance does not exclude model inference")
    config_path = root / "PREPARATION_CONFIG_COPY.json"
    config, started = read(config_path), read(root / "PREPARATION_STARTED.json")
    original_config = root / "preparation_config.json"
    # Preparation pins its original config bytes, but its archival JSON copy is
    # canonicalized. Formatting differences must not invalidate equal content.
    if (fineweb.file_sha256(original_config) != preparation["config_sha256"]
            or read(original_config) != config
            or started.get("config_sha256") != preparation["config_sha256"]
            or started.get("source_sha256") != preparation["preparation_source_sha256"]
            or started.get("no_model_inference") is not True
            or config["plan"]["sha256"] != preparation["plan_sha256"]):
        raise ValueError("FineWeb preparation configuration/source identity differs")
    for name, field in (("SOURCE_VERIFIED.json", "source_verification_sha256"),
                        ("DECODER_QUALIFIED.json", "decoder_qualification_sha256")):
        if fineweb.file_sha256(root / name) != preparation[field]:
            raise ValueError("FineWeb decoder/source preparation provenance changed")
    verified, decoder = read(root / "SOURCE_VERIFIED.json"), read(root / "DECODER_QUALIFIED.json")
    if (verified.get("revision") != training["revision"]
            or verified.get("provenance_sha256") != config["provenance"]["sha256"]
            or verified.get("prior_use_audit_sha256") != config["prior_use_audit"]["sha256"]
            or set(verified.get("sources", {})) != set(config["source_paths"])):
        raise ValueError("FineWeb source-role provenance differs")
    if (decoder.get("asset") != config["decoder"]
            or decoder.get("vocab_size") != fineweb.SOURCE_VOCAB
            or decoder.get("eot_id") != fineweb.SOURCE_EOT
            or decoder.get("utf8_and_whitespace_roundtrip") is not True
            or decoder.get("literal_eot_is_ordinary_text") is not True
            or decoder.get("document_roundtrip_rule") != "every accepted source document exact IDs"
            or decoder.get("complete_documents_roundtripped", 0) < 1
            or decoder.get("complete_tokens_roundtripped", 0) < 1
            or decoder.get("tokenizers_version") != fit.get("version")):
        raise ValueError("FineWeb decoder round-trip qualification differs")
    decoder_path = Path(decoder["asset"]["path"]).resolve()
    if not decoder_path.is_relative_to(root):
        raise ValueError("FineWeb pinned decoder asset escaped its data directory")
    fineweb.checked_file(decoder["asset"])
    return dict(preparation_complete_sha256=fineweb.file_sha256(root / "PREPARATION_COMPLETE.json"),
                roles_frozen_sha256=roles_sha, preparation=preparation)


def verify_training_data(cfg, panel):
    path = Path(cfg["data_path"]).resolve()
    if (cfg.get("corpus_kind") != KIND or path.name != "training_manifest.json"
            or fineweb.file_sha256(path) != cfg["data_sha256"]
            or Path(panel.path).resolve() != path.parent / "sealed_test"):
        raise ValueError("FineWeb training kind, manifest, or panel location differs")
    manifest = read(path)
    proof = verify_preparation(path.parent, manifest, panel.manifest,
                               training_sha256=cfg["data_sha256"], panel_sha256=panel.manifest_sha256)
    specs = [manifest[name] for name in ("tokenizer", "default_permutation", "dev_starts",
                                       "dev_lengths", "dev_document_indices")]
    specs += [split[name] for split in manifest["splits"].values() for name in ("tokens", "documents")]
    for spec in specs:
        asset(path.parent, spec)
    if manifest["tokenizer"]["sha256"] != manifest["tokenizer_sha256"]:
        raise ValueError("FineWeb fitted-tokenizer asset and identity differ")
    # The qualified loader verifies EOS, document/token integrity, all-target
    # development coverage, and the complete nonrecycled block permutation.
    with closing(fineweb.load_training_corpus(path, expected_manifest_sha256=cfg["data_sha256"])):
        pass
    identity = dict(manifest_sha256=cfg["data_sha256"], tokenizer_sha256=manifest["tokenizer_sha256"],
                    evaluation_policy_version=2, corpus_kind=KIND, **proof)
    return manifest, identity


def load_panel(path, *, expected_manifest_sha256):
    path = Path(path).resolve()
    root = path.parent if path.name == "manifest.json" else path
    manifest_path = root / "manifest.json"
    if fineweb.file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("FineWeb panel manifest hash mismatch")
    manifest = read(manifest_path)
    if manifest.get("corpus_kind") != KIND or manifest.get("role") != "sealed_confirmation":
        raise ValueError("FineWeb requires its own sealed confirmation manifest")
    training_path = root.parent / "training_manifest.json"
    verify_preparation(root.parent, read(training_path), manifest,
                       training_sha256=fineweb.file_sha256(training_path), panel_sha256=expected_manifest_sha256)
    for spec in manifest["splits"]["test"].values():
        if isinstance(spec, dict):
            asset(root, spec)
    panel = stories.load_test_panel(root, expected_manifest_sha256=expected_manifest_sha256)
    if panel.manifest.get("corpus_kind") != KIND or any(
            int(doc.token_ids[-1]) != manifest["eot_id"] for doc in panel.documents):
        raise ValueError("FineWeb test kind or document EOS integrity differs")
    return panel
