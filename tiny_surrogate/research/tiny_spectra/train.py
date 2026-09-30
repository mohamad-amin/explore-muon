"""Standalone tiny-surrogate training; no production training pipeline imports.

Each invocation is a fresh run. Interrupted runs are preserved, never resumed.
Character windows are paired across optimizers and are a common prefix across
batch sizes. Repeated training-corpus exposure is intentional and recorded.
"""
import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

import torch
from torch.nn import functional as F

from .data import load_corpus
from .model import GPT, ModelConfig
from .optim import make_optimizer, output_second_moments, snapshot_optimizer


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def save_model_snapshot(model, model_config, path, *, step, tokens):
    """Atomically save detached CPU forward state for evaluation, never resume.

    Statistic buffers are nonpersistent in this architecture and are excluded
    from state_dict. No optimizer state, parameter objects, RNG state, or second
    duplicate of the model tensors is stored. Saving performs no model forward.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite model snapshot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().to(device="cpu", copy=True)
             for name, tensor in model.state_dict().items()}
    digest = hashlib.sha256()
    for name, _ in model.named_parameters():
        digest.update(name.encode())
        digest.update(state[name].contiguous().numpy().tobytes())
    parameter_sha256 = digest.hexdigest()
    payload = dict(format="tiny_spectra_model_only_v1", resumable=False,
                   purpose="evaluation_only_no_optimizer_or_training_statistics",
                   step=int(step), tokens=int(tokens), model=state,
                   model_config=asdict(model_config), parameter_sha256=parameter_sha256)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # An incomplete previous attempt is preserved, not silently overwritten.
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return parameter_sha256


def amp(device, precision):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bf16" else contextlib.nullcontext())


def schedule_factor(step, total_steps, warmup_fraction, cooldown_fraction):
    warmup = max(1, math.ceil(total_steps * warmup_fraction))
    if step <= warmup:
        return step / warmup
    cooldown_start = math.floor(total_steps * (1 - cooldown_fraction))
    if step > cooldown_start:
        return (total_steps - step + 1) / max(1, total_steps - cooldown_start)
    return 1.0


@torch.no_grad()
def evaluate(model, corpus, split, starts, cfg, device):
    microbatch = cfg.get("evaluation_microbatch_sequences", cfg["microbatch_sequences"])
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("Evaluation microbatch must be a positive integer")
    was_training = model.training
    model.eval()
    sequence_losses = []
    sequence_counts = []
    masked = split == "val" and hasattr(corpus, "evaluation_batch")
    for chunk in starts.split(microbatch):
        if masked:
            x, y = corpus.evaluation_batch(split, chunk, cfg["seq_len"], device=device)
        else:
            x, y = corpus.batch(split, len(chunk), cfg["seq_len"], positions=chunk, device=device)
        with amp(device, cfg["precision"]):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
        if masked:
            counts = (y != -100).sum(dim=1)
            sequence_losses.append((loss.double().view_as(y).sum(dim=1) / counts).cpu())
            sequence_counts.append(counts.cpu())
        else:
            sequence_losses.append(loss.view_as(y).mean(dim=1).cpu())
    model.train(was_training)
    values = torch.cat(sequence_losses)
    if masked:
        counts = torch.cat(sequence_counts)
        return float((values * counts).sum() / counts.sum()), values
    return float(values.mean()), values


def training_backward(model, x, y, weight, cfg, graph=None):
    """One unchanged weighted training backward, optionally replayed as a graph."""
    if cfg.get("execution_backend", "eager") == "cudagraph":
        if graph is None:
            from .execution_graph import TrainingGraph
            graph = TrainingGraph(model, x, y, cfg["precision"])
        return graph.backward(x, y, weight), graph
    with amp(x.device, cfg["precision"]):
        loss = model(x, y)
    (loss * weight).backward()
    return loss.detach(), None


def run(cfg, out):
    execution_backend = cfg.get("execution_backend", "eager")
    if execution_backend != "eager":
        raise ValueError("Only eager execution is qualified for scientific training; the CUDA graph attempt failed fidelity")
    keep_model_every = cfg.get("keep_model_every", 0)
    if type(keep_model_every) is not int or keep_model_every < 0:
        raise ValueError("keep_model_every must be a nonnegative integer")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    atomic_json(out / "config.json", cfg)
    atomic_json(out / "status.json", dict(status="initializing", started_unix=started))
    device = torch.device(cfg.get("device", "cuda"))
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        # 45 GiB excludes nominal 48 GB cards whose usable memory is <48 GiB.
        if properties.total_memory >= 45 * 1024**3:
            raise RuntimeError("This study is restricted to GPUs strictly below nominal 48 GB VRAM")
        hardware = dict(name=properties.name, total_memory_bytes=properties.total_memory,
                        capability=list(torch.cuda.get_device_capability(device)))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        hardware = dict(name="CPU qualification", total_memory_bytes=None)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    corpus_kind = cfg.get("corpus_kind", "character")
    document_corpus = corpus_kind in ("tiny_stories_byte_bpe_v1", "fineweb_byte_bpe_v1")
    if corpus_kind == "tiny_stories_byte_bpe_v1":
        from .stories import load_training_corpus
        corpus = load_training_corpus(cfg["data_path"], expected_manifest_sha256=cfg["data_sha256"])
    elif corpus_kind == "fineweb_byte_bpe_v1":
        from .fineweb import load_training_corpus
        corpus = load_training_corpus(cfg["data_path"], expected_manifest_sha256=cfg["data_sha256"])
    elif corpus_kind == "character":
        corpus = load_corpus(cfg["data_path"], expected_sha256=cfg.get("data_sha256"))
    else:
        raise ValueError("Unknown training corpus kind")
    has_cov = cfg["method"] in ("pd", "ts", "spd", "sts")
    mc = ModelConfig(vocab_size=corpus.vocab_size, n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
                     n_head=cfg["n_head"], seq_len=cfg["seq_len"],
                     track_input_stats=has_cov, track_input_cov=has_cov,
                     cov_stride=cfg["cov_stride"], cov_decay=cfg["cov_ema"],
                     stats_decay=cfg.get("stats_ema", .99), stats_clock=cfg.get("stats_clock", "step"))
    model = GPT(mc).to(device)
    optimizer = make_optimizer(model, cfg, device)
    parameters = dict(model.named_parameters())
    body_names = list(model.body_parameters())
    body_name_set = set(body_names)
    aux_names = [name for name in parameters if name not in body_name_set]
    # The update capture is diagnostic only and does not change its computation.
    if hasattr(optimizer, "capture_parameters"):
        optimizer.capture_parameters = {parameters[body_names[0]], parameters[body_names[-1]]}
    total_sequences, remainder = divmod(cfg["total_tokens"], cfg["seq_len"])
    batch_sequences, batch_remainder = divmod(cfg["batch_tokens"], cfg["seq_len"])
    if remainder or batch_remainder or batch_sequences < 1:
        raise ValueError("Budget and effective batch must be positive multiples of context")
    total_steps = math.ceil(total_sequences / batch_sequences)
    generator = torch.Generator().manual_seed(cfg["seed"] + 1729)
    starts = corpus.window_positions("train", total_sequences, cfg["seq_len"], generator=generator)
    validation_starts = (corpus.evaluation_starts("val", cfg["seq_len"])
                         if hasattr(corpus, "evaluation_starts") else
                         torch.arange(0, len(corpus.tokens("val")) - cfg["seq_len"], cfg["seq_len"]))
    # Spread the fixed selection bank across the complete held-out split.
    count = min(len(validation_starts), cfg["validation_tokens"] // cfg["seq_len"])
    bank_indices = torch.linspace(0, len(validation_starts) - 1, count).round().long()
    validation_bank = validation_starts[bank_indices]
    if document_corpus:
        # A probe of examples actually exposed to training, paired across methods.
        train_bank = starts[:min(128, count)].clone()
    else:
        probe_gen = torch.Generator().manual_seed(91761)
        train_bank = corpus.window_positions("train", min(128, count), cfg["seq_len"], generator=probe_gen)
    metadata = dict(model_config=asdict(mc), n_parameters=model.num_parameters(),
                    body_parameters=sum(parameters[n].numel() for n in body_names),
                    auxiliary_parameters=sum(parameters[n].numel() for n in aux_names),
                    initial_parameter_sha256=model_hash(model),
                    train_window_sha256=tensor_hash(starts), validation_bank_sha256=tensor_hash(validation_bank),
                    corpus=corpus.manifest, hardware=hardware, host=socket.gethostname(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                    torch_version=torch.__version__, source_file=str(Path(__file__).resolve()),
                    total_steps=total_steps, repeated_corpus_exposure=cfg["total_tokens"] / len(corpus.tokens("train")),
                    covariance_clock=("pooled once per optimizer step" if mc.stats_clock == "step"
                                      else "EMA per collected training microforward"),
                    covariance_gram_precision="FP32; autocast disabled", started_unix=started)
    if document_corpus:
        lengths = corpus.evaluation_lengths("val", cfg["seq_len"])
        metadata.update(data_manifest_sha256=cfg["data_sha256"],
                        validation_bank_lengths_sha256=tensor_hash(lengths[bank_indices]),
                        validation_bank_target_count=int(lengths[bank_indices].sum()),
                        full_development_target_count=int(lengths.sum()),
                        evaluation_policy="all within-document targets; masked right padding; token-weighted NLL",
                        train_probe_policy="first paired exposed training blocks")
    if keep_model_every:
        metadata["model_snapshot_policy"] = dict(
            every=keep_model_every, steps="0, 1, every N, final",
            path="models/stepNNNNNN.pt", format="tiny_spectra_model_only_v1",
            resumable=False)
    atomic_json(out / "metadata.json", metadata)
    torch.save(dict(train=starts, validation=validation_bank, train_probe=train_bank), out / "windows.pt")
    records = []
    metrics = (out / "metrics.jsonl").open("x", buffering=1)
    initial_val, _ = evaluate(model, corpus, "val", validation_bank, cfg, device)
    records.append(dict(step=0, tokens=0, validation_nll=initial_val, elapsed_seconds=time.time()-started))
    metrics.write(json.dumps(records[-1]) + "\n")
    print(json.dumps(dict(event="initial", **records[-1], parameters=model.num_parameters(), hardware=hardware)), flush=True)
    if keep_model_every:
        save_model_snapshot(model, mc, out / "models/step000000.pt", step=0, tokens=0)
    science_start = time.perf_counter()
    training_seconds = 0.0
    for step in range(1, total_steps + 1):
        factor = schedule_factor(step, total_steps, cfg["warmup_fraction"], cfg["cooldown_fraction"])
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * factor * group.get("lr_scale", 1.0)
        begin = (step - 1) * batch_sequences
        batch_starts = starts[begin:begin + batch_sequences]
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        if cfg["method"] in ("ts", "sts") and (step-1) % cfg["root_refresh"] == 0:
            selected = batch_starts[:cfg["out_sequences"]]
            x, y = corpus.batch("train", len(selected), cfg["seq_len"], positions=selected, device=device)
            label_generator = torch.Generator(device=device).manual_seed(cfg["seed"] * 1000003 + step * 97)
            stats = output_second_moments(model, x, y, source="gn", precision=cfg["precision"],
                                          generator=label_generator)
            optimizer.update_output_statistics(stats, cfg["out_ema"])
        model.begin_step_stats()
        accumulated_loss = torch.zeros((), device=device)
        for chunk in batch_starts.split(cfg["microbatch_sequences"]):
            x, y = corpus.batch("train", len(chunk), cfg["seq_len"], positions=chunk, device=device)
            weight = len(chunk) / len(batch_starts)
            loss, _ = training_backward(model, x, y, weight, cfg)
            accumulated_loss += loss.float() * weight
        model.finish_step_stats(ema=cfg["cov_ema"] if mc.stats_clock == "step" else None)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        report = step == 1 or step % cfg["eval_every"] == 0 or step == total_steps
        if report:
            before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_step = time.perf_counter() - tick
        training_seconds += elapsed_step
        record = dict(step=step, tokens=min(begin + len(batch_starts), total_sequences) * cfg["seq_len"],
                      train_nll=float(accumulated_loss), lr=cfg["lr"]*factor,
                      gradient_norm_before_clip=float(norm), step_seconds=elapsed_step,
                      elapsed_seconds=time.perf_counter()-science_start)
        if not math.isfinite(record["train_nll"]):
            raise FloatingPointError("Nonfinite training loss")
        if report:
            for group_name, names in (("body", body_names), ("auxiliary", aux_names)):
                record[group_name + "_parameter_norm"] = float(sum(parameters[n].detach().square().sum() for n in names).sqrt())
                record[group_name + "_actual_delta_norm"] = float(sum((parameters[n].detach()-before[n]).square().sum() for n in names).sqrt())
            del before
            record["validation_nll"], _ = evaluate(model, corpus, "val", validation_bank, cfg, device)
            record["train_probe_nll"], _ = evaluate(model, corpus, "train", train_bank, cfg, device)
            if hasattr(optimizer, "last_updates"):
                record["captured_direction_norms"] = {name: float(optimizer.last_updates[p].norm())
                    for name, p in parameters.items() if p in optimizer.last_updates}
            print(json.dumps(record), flush=True)
        records.append(record)
        metrics.write(json.dumps(record) + "\n")
        if keep_model_every and (step == 1 or step % keep_model_every == 0 or step == total_steps):
            save_model_snapshot(model, mc, out / "models" / f"step{step:06d}.pt",
                                step=step, tokens=record["tokens"])
        atomic_json(out / "status.json", dict(status="running", step=step, tokens=record["tokens"],
                                             total_steps=total_steps, updated_unix=time.time()))
    final_full, per_sequence = evaluate(model, corpus, "val", validation_starts, cfg, device)
    final_validation = dict(starts=validation_starts, sequence_nll=per_sequence)
    document_results = None
    if document_corpus:
        counts = corpus.evaluation_lengths("val", cfg["seq_len"])
        document_ids = corpus.evaluation_document_indices("val", cfg["seq_len"])
        final_validation.update(target_counts=counts, document_indices=document_ids)
        docs = corpus.development_documents
        sums = torch.zeros(len(docs), dtype=torch.float64).scatter_add_(0, document_ids, per_sequence.double() * counts)
        totals = torch.zeros(len(docs), dtype=torch.long).scatter_add_(0, document_ids, counts)
        document_results = [dict(identity=doc["identity"], targets=int(totals[i]),
                                 nll=float(sums[i] / totals[i]) if totals[i] else None)
                            for i, doc in enumerate(docs)]
        atomic_json(out / "development_documents.json", dict(documents=document_results,
                    weighting="token count", total_targets=int(totals.sum())))
    torch.save(final_validation, out / "final_validation.pt")
    snapshot = snapshot_optimizer(model, optimizer)
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    optimizer=snapshot, config=cfg, metadata=metadata), out / "final.pt")
    summary = dict(status="complete", final_validation_nll=records[-1]["validation_nll"],
                   full_validation_nll=final_full, train_probe_nll=records[-1]["train_probe_nll"],
                   steps=total_steps, tokens=cfg["total_tokens"], training_seconds=training_seconds,
                   total_seconds=time.time()-started,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0,
                   tensor_memory=snapshot["tensor_memory"],
                   n_parameters=model.num_parameters(), hardware=hardware,
                   initial_parameter_sha256=metadata["initial_parameter_sha256"],
                   train_window_sha256=metadata["train_window_sha256"])
    if document_results is not None:
        groups = []
        for group_id in range(4):
            members = [d for d in document_results if int(d["identity"][:2], 16) % 4 == group_id and d["targets"]]
            targets = sum(d["targets"] for d in members)
            groups.append(dict(group_id=group_id, documents=len(members), targets=targets,
                               nll=sum(d["nll"] * d["targets"] for d in members) / targets if targets else None))
        summary.update(selection_metric=cfg.get("selection_metric", "bank"),
                       full_development_target_count=sum(d["targets"] for d in document_results),
                       development_groups=groups,
                       development_group_rule="first SHA256 byte modulo4; fixed before model scores")
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", dict(**summary, finished_unix=time.time()))
    metrics.close()
    print(json.dumps(dict(event="complete", **summary)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    # Pre-existing results are never altered, even on a failed launch.
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    try:
        run(cfg, args.out)
    except Exception as error:
        path = Path(args.out)
        if path.exists():
            atomic_json(path / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(),
                                                   failed_unix=time.time()))
            atomic_json(path / "status.json", dict(status="failed", error=repr(error), failed_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
