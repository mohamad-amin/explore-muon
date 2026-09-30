"""One paired CUDA-graph fidelity/cost qualification; no test-panel access."""
import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.torch_version import TorchVersion

from .fineweb import load_training_corpus
from .model import GPT, ModelConfig
from .optim import make_optimizer, output_second_moments, snapshot_optimizer
from .qualify_d import digest, read, verify_sources
from .train import atomic_json, evaluate, model_hash, schedule_factor, tensor_hash, training_backward


def compare(left, right, plan, path="state"):
    """Finite, structure-aware comparison; all thresholds precede measurement."""
    result = dict(tensors=0, elements=0, maximum_absolute_error=0., maximum_tolerance_ratio=0.)

    def visit(a, b, name):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("Tensor shape/dtype differs: " + name)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Nonfinite tensor: " + name)
            result["tensors"] += 1
            result["elements"] += a.numel()
            if torch.equal(a, b):
                return
            if not a.is_floating_point():
                raise ValueError("Integer state differs: " + name)
            error = (a.double()-b.double()).abs()
            bound = plan["floating_tensor_atol"] + plan["floating_tensor_rtol"]*a.double().abs()
            maximum = float(error.max())
            ratio = float((error/bound).max())
            result["maximum_absolute_error"] = max(result["maximum_absolute_error"], maximum)
            result["maximum_tolerance_ratio"] = max(result["maximum_tolerance_ratio"], ratio)
            if ratio > 1:
                raise ValueError(f"Tensor fidelity failed: {name}; max_abs={maximum}; max_tolerance_ratio={ratio}")
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError("State keys differ: " + name)
            for key in a:
                visit(a[key], b[key], name+"/"+str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError("State sequence differs: " + name)
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, name+"/"+str(i))
        elif isinstance(a, float):
            if not isinstance(b, (int, float)) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Invalid scalar: " + name)
            if abs(a-b) > plan["floating_tensor_atol"]+plan["floating_tensor_rtol"]*abs(a):
                raise ValueError("Scalar state differs: " + name)
        elif type(a) is not type(b) or a != b:
            raise ValueError("State value differs: " + name)

    visit(left, right, path)
    return result


def cpu_parameters(model):
    return {name:p.detach().cpu().clone() for name, p in model.named_parameters()}


def numerical_state(model, optimizer):
    snapshot = snapshot_optimizer(model, optimizer)
    return dict(model={name:value.detach().cpu().clone() for name,value in model.state_dict().items()},
                optimizer={name:snapshot[name] for name in ("state", "external", "model_statistics", "param_groups")})


def checked_capture(model,x,y,precision):
    """Prove warmup/capture is observationally neutral before its first replay."""
    from .execution_graph import TrainingGraph
    parameters = cpu_parameters(model)
    buffers = {name:b.detach().cpu().clone() for name,b in model.named_buffers()}
    gradients = {name:(p.grad,None if p.grad is None else p.grad.detach().cpu().clone())
                 for name,p in model.named_parameters()}
    modes = [(m,m.training,getattr(m,"collect_stats",None)) for m in model.modules()]
    cpu_rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state(x.device)
    python_rng,numpy_rng = random.getstate(),np.random.get_state()
    graph = TrainingGraph(model,x,y,precision)
    for name,value in cpu_parameters(model).items():
        if not torch.equal(value,parameters[name]):
            raise ValueError("Capture changed a parameter")
    for name,value in model.named_buffers():
        if not torch.equal(value.detach().cpu(),buffers[name]):
            raise ValueError("Capture changed a statistic or other buffer: "+name)
    for name,parameter in model.named_parameters():
        original,value = gradients[name]
        if parameter.grad is not original or (value is not None and not torch.equal(parameter.grad.cpu(),value)):
            raise ValueError("Capture changed a gradient value or its None/storage identity")
    if any(m.training != mode or getattr(m,"collect_stats",None) != collecting for m,mode,collecting in modes):
        raise ValueError("Capture changed a training/collection mode")
    if (not torch.equal(cpu_rng,torch.get_rng_state())
            or not torch.equal(cuda_rng,torch.cuda.get_rng_state(x.device))
            or random.getstate() != python_rng
            or any(not np.array_equal(a,b) for a,b in zip(np.random.get_state(),numpy_rng))):
        raise ValueError("Capture changed an RNG state")
    return graph


def make_pair(cfg, corpus, device):
    pair = {}
    for backend in ("eager", "cudagraph"):
        tick = time.perf_counter()
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        geometry = cfg["method"] in ("pd", "ts", "spd")
        mc = ModelConfig(vocab_size=corpus.vocab_size, n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
                         n_head=cfg["n_head"], seq_len=cfg["seq_len"], track_input_stats=geometry,
                         track_input_cov=geometry, cov_stride=cfg["cov_stride"], cov_decay=cfg["cov_ema"],
                         stats_decay=cfg["stats_ema"], stats_clock=cfg["stats_clock"])
        model = GPT(mc).to(device)
        config = dict(cfg, execution_backend=backend)
        optimizer = make_optimizer(model, config, device)
        body = model.body_parameters()
        if hasattr(optimizer, "capture_parameters"):
            optimizer.capture_parameters = {body[next(iter(body))], body[next(reversed(body))]}
        torch.cuda.synchronize(device)
        pair[backend] = dict(model=model, optimizer=optimizer, cfg=config, graph=None,
                             initialization_seconds=time.perf_counter()-tick, training_seconds=0.,
                             evaluation_seconds=0., setup_seconds=0., previous=cpu_parameters(model),
                             initial_parameter_sha256=model_hash(model), model_config=asdict(mc))
    return pair


def train_step(item, corpus, positions, step, device):
    """Same operations as train.run; diagnostic transfers are timed separately."""
    model, optimizer, cfg = item["model"], item["optimizer"], item["cfg"]
    factor = schedule_factor(min(step,21), 21, cfg["warmup_fraction"], cfg["cooldown_fraction"])
    for group in optimizer.param_groups:
        group["lr"] = cfg["lr"]*factor*group.get("lr_scale", 1.)
    optimizer.zero_grad(set_to_none=cfg["execution_backend"] == "eager")
    torch.cuda.synchronize(device)
    tick = time.perf_counter()
    if cfg["method"] == "ts" and (step-1) % cfg["root_refresh"] == 0:
        selected = positions[:cfg["out_sequences"]]
        x,y = corpus.batch("train",len(selected),512,positions=selected,device=device)
        generator = torch.Generator(device=device).manual_seed(cfg["seed"]*1000003+step*97)
        optimizer.update_output_statistics(output_second_moments(
            model,x,y,source="gn",precision=cfg["precision"],generator=generator),cfg["out_ema"])
    model.begin_step_stats()
    accumulated = torch.zeros((),device=device)
    setup_audit_seconds = 0.
    for chunk in positions.split(4):
        x,y = corpus.batch("train",len(chunk),512,positions=chunk,device=device)
        weight = len(chunk)/len(positions)
        if cfg["execution_backend"] == "cudagraph" and item["graph"] is None:
            audit_tick = time.perf_counter()
            item["graph"] = checked_capture(model,x,y,cfg["precision"])
            setup_audit_seconds = time.perf_counter()-audit_tick-item["graph"].setup_seconds
        loss,item["graph"] = training_backward(model,x,y,weight,cfg,item["graph"])
        accumulated += loss.float()*weight
    model.finish_step_stats(ema=None)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter()-tick-setup_audit_seconds
    if item["graph"] is not None and item["setup_seconds"] == 0.:
        item["setup_seconds"] = item["graph"].setup_seconds
        seconds -= item["setup_seconds"]
    # Read the raw gradient before clipping, outside the measured training time.
    gradients = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()}
    tick = time.perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    seconds += time.perf_counter()-tick
    if not math.isfinite(float(accumulated)) or seconds <= 0:
        raise ValueError("Invalid training loss or measured time")
    state = numerical_state(model,optimizer)
    parameters = cpu_parameters(model)
    deltas = {name:value-item["previous"][name] for name,value in parameters.items()}
    item["previous"] = parameters
    return dict(loss=float(accumulated),norm=float(norm),seconds=seconds,
                gradients=gradients,deltas=deltas,state=state)


def check_loss(a,b,plan):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > plan["loss_absolute_tolerance"]:
        raise ValueError(f"Loss fidelity failed: {a} versus {b}")
    return abs(a-b)


def paired_evaluation(pair,corpus,starts,train_probe,plan,device):
    values = {}
    for backend,item in pair.items():
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        mean,windows = evaluate(item["model"],corpus,"val",starts,item["cfg"],device)
        probe = (evaluate(item["model"],corpus,"train",train_probe,item["cfg"],device)[0]
                 if train_probe is not None else None)
        torch.cuda.synchronize(device)
        item["evaluation_seconds"] += time.perf_counter()-tick
        values[backend] = (mean,windows,probe)
    a,b = values["eager"],values["cudagraph"]
    error = check_loss(a[0],b[0],plan)
    if not torch.isfinite(a[1]).all() or not torch.isfinite(b[1]).all():
        raise ValueError("Nonfinite development loss")
    maximum = float((a[1]-b[1]).abs().max())
    if maximum > plan["loss_absolute_tolerance"]:
        raise ValueError("Per-window development loss fidelity failed")
    if a[2] is not None:
        check_loss(a[2],b[2],plan)
    return dict(mean_absolute_error=error,maximum_window_absolute_error=maximum),values


def method_benchmark(cfg,corpus,plan,out,device,baseline):
    out.mkdir(parents=True,exist_ok=False)
    tick = time.perf_counter()
    generator = torch.Generator().manual_seed(cfg["seed"]+1729)
    starts = corpus.window_positions("train",21*512+69,512,generator=generator)
    full_starts = corpus.evaluation_starts("val",512)
    original = Path(plan["reference_cohort"])/"runs"/cfg["run_id"]
    original_metadata = read(original/"metadata.json")
    if (tensor_hash(starts[:21*512]) != original_metadata["train_window_sha256"]
            or tensor_hash(full_starts) != original_metadata["validation_bank_sha256"]
            or int(corpus.evaluation_lengths("val",512).sum()) != plan["full_development_targets"]):
        raise ValueError("Benchmark population differs from the frozen numerical smoke")
    pair = make_pair(cfg,corpus,device)
    for item in pair.values():
        if item["initial_parameter_sha256"] != original_metadata["initial_parameter_sha256"]:
            raise ValueError("Benchmark initialization differs from the frozen smoke")
    setup_and_stream_seconds = time.perf_counter()-tick
    evaluations = {}
    evaluations["0"],_ = paired_evaluation(pair,corpus,full_starts,None,plan,device)
    rows = []
    for step in range(1,23):
        positions = starts[(step-1)*512:step*512] if step <= 21 else starts[21*512:]
        measured = {backend:train_step(item,corpus,positions,step,device) for backend,item in pair.items()}
        a,b = measured["eager"],measured["cudagraph"]
        if cfg["method"] in ("pd","ts","spd"):
            expected_total = step*128 if step <= 21 else 2706
            expected_step = 128 if step <= 21 else 18
            for value in measured.values():
                statistics = value["state"]["optimizer"]["model_statistics"]
                if len(statistics) != 48 or any(int(s["_total_forwards"]) != expected_total
                        or int(s["_step_forwards"]) != expected_step for s in statistics.values()):
                    raise ValueError("Training-only statistic clock or partial-batch count differs")
        if (a["norm"] > cfg["grad_clip"]) != (b["norm"] > cfg["grad_clip"]):
            raise ValueError("Gradient clipping decisions differ")
        row = dict(step=step,partial_batch=step==22,sequences=len(positions),
                   train_loss_absolute_error=check_loss(a["loss"],b["loss"],plan),
                   gradients=compare(a["gradients"],b["gradients"],plan,"raw_gradients"),
                   actual_parameter_deltas=compare(a["deltas"],b["deltas"],plan,"parameter_deltas"),
                   state=compare(a["state"],b["state"],plan),
                   eager_seconds=a["seconds"],graph_seconds=b["seconds"])
        rows.append(row)
        atomic_json(out/"progress.json",dict(status="running",step=step,comparisons_passed=True))
        if step <= 21:
            for backend,item in pair.items():
                item["training_seconds"] += measured[backend]["seconds"]
        if step in (1,21):
            evaluations[str(step)],values = paired_evaluation(pair,corpus,full_starts,starts[:128],plan,device)
        if step == 21:
            # The modified harness must itself reproduce the preserved trainer.
            with torch.serialization.safe_globals([TorchVersion]):
                reference = torch.load(original/"final.pt",map_location="cpu",weights_only=True)
            reference_state = dict(model=reference["model"],optimizer={name:reference["optimizer"][name]
                for name in ("state","external","model_statistics","param_groups")})
            reference_match = compare(reference_state,a["state"],plan,"frozen_eager_reference")
            saved = torch.load(original/"final_validation.pt",map_location="cpu",weights_only=True)
            if float((saved["sequence_nll"]-values["eager"][1]).abs().max()) > plan["loss_absolute_tolerance"]:
                raise ValueError("New eager evaluator differs from the preserved eager reference")
            for backend in pair:
                torch.save(measured[backend]["state"],out/(backend+"_step21.pt"))
            # Include the trainer's separate final full-development evaluation.
            evaluations["final_reconstruction"],_ = paired_evaluation(pair,corpus,full_starts,None,plan,device)
            del reference,reference_state
        print(json.dumps(dict(method=cfg["method"],step=step,paired_fidelity_passed=True)),flush=True)
    graph = pair["cudagraph"]["graph"]
    if graph.replay_count != 2705 or graph.fallback_count != 1:
        raise ValueError("Expected 2688+17 graph forwards and one partial eager fallback")
    candidate = pair["cudagraph"]
    windows = plan["full_development_windows"]
    work_ratio = (49*windows+47*128)/(4*windows+2*128)
    nontraining = max(baseline["total_seconds"]-baseline["training_seconds"],
                     setup_and_stream_seconds+candidate["evaluation_seconds"])
    projected = (candidate["training_seconds"]*368/21 + candidate["setup_seconds"]
                 + nontraining*work_ratio + rows[-1]["graph_seconds"])
    result = dict(method=cfg["method"],paired_fidelity_passed=True,rows=rows,evaluations=evaluations,
        capture_restoration_verified=True,
        frozen_eager_reference=reference_match,training_seconds=candidate["training_seconds"],
        capture_setup_seconds=candidate["setup_seconds"],nontraining_overhead_floor_seconds=nontraining,
        partial_fallback_seconds=rows[-1]["graph_seconds"],projected_full_run_seconds=projected,
        graph_replay_count=graph.replay_count,graph_fallback_count=graph.fallback_count,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        note="Paired diagnostics/transfers excluded from training timers; full initialization/evaluation floor retained.")
    atomic_json(out/"results.json",result)
    atomic_json(out/"progress.json",dict(status="complete",paired_fidelity_passed=True))
    return result


def run(cohort):
    cohort = Path(cohort).resolve()
    if (cohort/"EXECUTION_STARTED.json").exists():
        raise FileExistsError("Preserve the first execution attempt; do not rerun")
    verify_sources(cohort)
    preflight = read(cohort/"PREFLIGHT.json")
    if (preflight["plan_sha256"] != digest(cohort/"PLAN.json")
            or preflight["source_manifest_sha256"] != digest(cohort/"source_manifest.json")):
        raise ValueError("Frozen graph benchmark plan/source identity differs")
    plan = read(cohort/"PLAN.json")
    reference = Path(plan["reference_cohort"])
    verify_sources(reference)
    if digest(reference/"report/results.json") != preflight["reference_report_sha256"]:
        raise ValueError("Preserved numerical qualification report changed")
    original = read(reference/"report/results.json")
    for baseline in original["runs"]:
        root = reference/"runs"/baseline["run_id"]
        for name,sha in baseline["artifact_sha256"].items():
            if digest(root/name) != sha:
                raise ValueError("Preserved numerical reference artifact changed: "+str(root/name))
    configs = {read(p)["method"]:read(p) for p in (reference/"configs").glob("*.json")}
    if set(configs) != set(plan["methods"]):
        raise ValueError("Reference methods differ")
    for name,sha in preflight["reference_config_hashes"].items():
        if digest(reference/"configs"/name) != sha:
            raise ValueError("Reference configuration changed")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    if "A4000" not in properties.name or properties.total_memory >= 45*1024**3:
        raise ValueError("The single declared sub-48GB A4000 is required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    atomic_json(cohort/"EXECUTION_STARTED.json",dict(started_unix=time.time(),
        plan_sha256=digest(cohort/"PLAN.json"),hardware=dict(name=properties.name,total_memory_bytes=properties.total_memory)))
    try:
        corpus = load_training_corpus(plan["data_path"],expected_manifest_sha256=plan["training_manifest_sha256"])
        rows = []
        try:
            for method in plan["methods"]:
                torch.cuda.reset_peak_memory_stats(device)
                baseline = next(row for row in original["runs"] if row["method"] == method)
                rows.append(method_benchmark(configs[method],corpus,plan,cohort/"runs"/method,device,baseline))
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            corpus.close()
        forecast = 8*sum(row["projected_full_run_seconds"] for row in rows)/3600
        result = dict(all_five_paired_fidelity_passed=True,maximum40_forecast_gpu_hours=forecast,
            resource_gate_passed=forecast <= plan["screen_gpu_hour_cap"],methods=rows,
            scientific_screen_launched=False,scientific_ranking_claim=False,sealed_test_scored=False,
            decision="Execution objection resolved; a separate screen decision is still required." if forecast<=8 else
                     "Execution attempt closed: the unchanged resource gate still fails.")
        atomic_json(cohort/"RESULTS.json",result)
        atomic_json(cohort/"EXECUTION_COMPLETE.json",dict(completed_unix=time.time(),result_sha256=digest(cohort/"RESULTS.json")))
    except BaseException as error:
        atomic_json(cohort/"EXECUTION_FAILURE.json",dict(error=repr(error),traceback=traceback.format_exc(),
                    failed_unix=time.time(),attempt_closed=True))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",type=Path)
    run(parser.parse_args().cohort)
