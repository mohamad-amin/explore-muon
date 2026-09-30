"""Complete-family development readout for Candidate D; never scores a model."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import statistics

import numpy as np
import torch
from torch.torch_version import TorchVersion

from .ordering_d import METHODS,SEEDS,read,sha,verify,write,classify_exit,make_script
from .qualify_d import fp32_ema_weight,tensors
from .cohort import freeze


class NotReady(ValueError):
    pass


def population(cfg):
    path = Path(cfg["data_path"])
    if sha(path) != cfg["data_sha256"]:
        raise ValueError("Training/development manifest changed")
    manifest = read(path)
    result = {}
    for key,name in (("starts","dev_starts"),("target_counts","dev_lengths"),("document_indices","dev_document_indices")):
        spec = manifest[name]
        asset = (path.parent/spec["path"]).resolve()
        if not asset.is_relative_to(path.parent.resolve()) or asset.stat().st_size != spec["bytes"] or sha(asset) != spec["sha256"]:
            raise ValueError("Development population asset changed")
        result[key] = torch.from_numpy(np.fromfile(asset,dtype="<i8").copy())
    spec = manifest["splits"]["val"]["documents"]
    asset = path.parent/spec["path"]
    if sha(asset) != spec["sha256"]:
        raise ValueError("Development document identities changed")
    return result,read(asset)


def required_configs(cohort,stage):
    verify(cohort,"initial")
    tasks = read(cohort/"tasks_initial.json")
    if stage == "final" and (cohort/"EDGE_PREFLIGHT.json").exists():
        verify(cohort,"edges")
        edge = read(cohort/"EDGE_PREFLIGHT.json")
        if sha(cohort/"report_initial/results.json") != edge["initial_report_sha256"]:
            raise ValueError("The pre-edge initial readout changed")
        tasks += read(cohort/"tasks_edges.json")
    configs = [read(arm["config"]) for group in tasks for arm in group]
    keys = [(c["method"],c["seed"],c["lr"]) for c in configs]
    if len(set(keys)) != len(keys) or len(configs) not in range(30,41,2):
        raise ValueError("Incomplete or duplicate comparison family")
    # Check only terminal classifications until every requested arm is ready.
    incomplete = []
    for cfg in configs:
        record = cohort/"runs"/cfg["run_id"]/"ARM_EXECUTION.json"
        if not record.is_file(): incomplete.append(cfg["run_id"])
        elif read(record).get("status") not in ("complete","numerical_instability"):
            raise ValueError("Operational failure invalidates the ordering readout")
    if (cohort/"HALT.json").exists():
        raise ValueError("Cohort has a preserved operational failure")
    if incomplete:
        raise NotReady(f"Wait for the complete declared family; {len(incomplete)} arms remain")
    if stage == "final" and (cohort/"EDGE_PREFLIGHT.json").exists():
        initial = read(cohort/"report_initial/results.json")
        for row in initial["rows"]:
            for name,expected in row["artifact_sha256"].items():
                if sha(cohort/"runs"/row["run_id"]/name) != expected:
                    raise ValueError("An initial-stage artifact changed after rate selection")
    return configs


def select_rates(rows):
    """One joint rate per method; an unstable paired recipe cannot win."""
    selected = {}
    for method in METHODS:
        members = [r for r in rows if r["method"]==method]
        rates = sorted({r["lr"] for r in members})
        candidates = []
        for lr in rates:
            pair = [r for r in members if r["lr"]==lr]
            if sorted(r["seed"] for r in pair) != list(SEEDS):
                raise ValueError("Every tested rate needs both fixed seeds")
            complete = all(r["status"]=="complete" for r in pair)
            candidates.append(dict(lr=lr,stable_both_seeds=complete,
                mean_nll=statistics.mean(r["full_development_nll"] for r in pair) if complete else None,
                by_seed={str(r["seed"]):r["full_development_nll"] for r in pair},
                run_ids=[r["run_id"] for r in pair]))
        stable = [c for c in candidates if c["stable_both_seeds"]]
        winner = min(stable,key=lambda c:(c["mean_nll"],c["lr"])) if stable else None
        minima = [c["lr"] for c in stable if c["mean_nll"]==winner["mean_nll"]] if winner else []
        boundary = [lr for lr in minima if lr in (rates[0],rates[-1])]
        edge = rates[0]/2 if boundary==[rates[0]] else rates[-1]*2 if boundary==[rates[-1]] else None
        selected[method] = dict(candidates=candidates,selected=winner,minimizing_rates=minima,
            bracket_closed=bool(winner and len(rates)>=3 and not boundary),
            ambiguous_both_boundaries=len(boundary)==2,proposed_single_outward_lr=edge)
    return selected


def ordering_gates(selected):
    pairs = []
    for earlier,later in zip(METHODS,METHODS[1:]):
        a,b = selected[earlier]["selected"],selected[later]["selected"]
        differences = {str(s):b["by_seed"][str(s)]-a["by_seed"][str(s)] for s in SEEDS} if a and b else None
        mean = statistics.mean(differences.values()) if differences else None
        pairs.append(dict(earlier=earlier,later=later,differences=differences,mean_difference=mean,
            negative_in_both_seeds=bool(differences and all(x<0 for x in differences.values())),
            mean_advantage_at_least_005=bool(mean is not None and mean<=-.005)))
    passed = (all(s["bracket_closed"] for s in selected.values()) and
              all(p["negative_in_both_seeds"] and p["mean_advantage_at_least_005"] for p in pairs))
    return pairs,passed


def plot_readout(result,out):
    """Plot all declared LR responses and the jointly selected development curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=dict(adamw="AdamW",muon="Muon",pd="PD",ts="TS",spd="SOAP-PD")
    colors=dict(zip(METHODS,plt.get_cmap("tab10").colors))
    figure,axes=plt.subplots(1,3,figsize=(15,4.3))
    for method in METHODS:
        choice=result["joint_lr_selections"][method]
        candidates=choice["candidates"]
        for seed in SEEDS:
            y=[c["by_seed"][str(seed)] if c["by_seed"][str(seed)] is not None else np.nan for c in candidates]
            axes[0].plot([c["lr"] for c in candidates],y,":o",color=colors[method],alpha=.4,markersize=3)
        axes[0].plot([c["lr"] for c in candidates],
                     [c["mean_nll"] if c["mean_nll"] is not None else np.nan for c in candidates],
                     "-o",color=colors[method],label=labels[method],markersize=4)
        winner=choice["selected"]
        if winner is None:continue
        selected=[r for r in result["rows"] if r["run_id"] in winner["run_ids"]]
        curves=[]
        for row in selected:
            x=np.array([e["tokens"] for e in row["evaluations"]])/1e6
            y=np.array([e["validation_nll"] for e in row["evaluations"]])
            curves.append(y)
            axes[1].plot(x,y,":",color=colors[method],alpha=.4)
        axes[1].plot(x,np.mean(curves,axis=0),color=colors[method],label=labels[method])
        if method!="muon" and result["joint_lr_selections"]["muon"]["selected"] is not None:
            baseline=result["joint_lr_selections"]["muon"]["selected"]["run_ids"]
            differences=[]
            for row in selected:
                control=next(r for r in result["rows"] if r["run_id"] in baseline and r["seed"]==row["seed"])
                difference=np.array([e["validation_nll"] for e in row["evaluations"]])-np.array(
                    [e["validation_nll"] for e in control["evaluations"]])
                differences.append(difference)
                axes[2].plot(x,difference,":",color=colors[method],alpha=.4)
            axes[2].plot(x,np.mean(differences,axis=0),color=colors[method],label=labels[method])
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Body learning rate",ylabel="Final full-development NLL",title="All tested rates")
    axes[1].set(xlabel="Training targets (millions)",ylabel="Full-development NLL",title="One joint LR per method")
    axes[2].axhline(0,color="black",linewidth=.7)
    axes[2].set(xlabel="Training targets (millions)",ylabel="NLL difference from Muon",title="Paired differences")
    for axis in axes:
        axis.grid(alpha=.15)
        axis.spines[["top","right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    figure.suptitle("Candidate D — development only; dotted lines are seeds, solid lines are their mean")
    figure.tight_layout()
    for suffix in ("png","pdf"):
        figure.savefig(out/("ordering_curves."+suffix),dpi=170)
    plt.close(figure)


def report(cohort,stage):
    cohort = Path(cohort).resolve()
    out = cohort/("report_"+stage)
    if out.exists():
        raise FileExistsError("Preserve the completed readout")
    plan = read(cohort/"PLAN.json")
    configs = required_configs(cohort,stage)
    pop,documents = population(configs[0])
    if len(pop["starts"]) != plan["full_development_windows"] or int(pop["target_counts"].sum()) != plan["full_development_targets"]:
        raise ValueError("Development denominator differs from the frozen plan")
    weights = dict(input_cov_weight=fp32_ema_weight(.998,46994),input_weight=fp32_ema_weight(.99,46994))
    rows,paired = [],{}
    for cfg in configs:
        root = cohort/"runs"/cfg["run_id"]
        execution = read(root/"ARM_EXECUTION.json")
        if (execution["config_sha256"] != sha(cohort/"configs"/(cfg["run_id"]+".json"))
                or execution["source_manifest_sha256"] != sha(cohort/"source_manifest.json")
                or read(root/"config.json") != cfg
                or classify_exit(root,execution["returncode"]) != execution["status"]):
            raise ValueError("Run terminal/configuration identity differs")
        meta = read(root/"metadata.json")
        hw = meta["hardware"]
        if ("A4000" not in hw["name"] or not 0<hw["total_memory_bytes"]<45*1024**3
                or meta["data_manifest_sha256"] != plan["training_manifest_sha256"]
                or meta["n_parameters"] != 4861056 or meta["total_steps"] != 368
                or meta["model_config"]["stats_clock"] != "microforward"
                or meta["covariance_gram_precision"] != "FP32; autocast disabled"
                or Path(meta["source_file"]).resolve() != cohort/"frozen/research/tiny_spectra/train.py"):
            raise ValueError("Runtime data, architecture, hardware or source differs")
        identity = tuple(meta[k] for k in ("initial_parameter_sha256","train_window_sha256","validation_bank_sha256",
                                           "validation_bank_lengths_sha256","data_manifest_sha256"))
        if cfg["seed"] in paired and paired[cfg["seed"]] != identity:
            raise ValueError("Methods/rates do not have paired initialization and data")
        paired[cfg["seed"]] = identity
        row = dict(run_id=cfg["run_id"],method=cfg["method"],seed=cfg["seed"],lr=cfg["lr"],
                   status=execution["status"],full_development_nll=None,hardware=hw)
        if row["status"] == "numerical_instability":
            row["failure"] = read(root/"failure.json")
            row["artifact_sha256"] = {name:sha(root/name) for name in ("ARM_EXECUTION.json","failure.json","metadata.json","metrics.jsonl","config.json")}
            rows.append(row)
            continue
        summary = read(root/"summary.json")
        metrics = [json.loads(line) for line in (root/"metrics.jsonl").read_text().splitlines()]
        if (summary.get("status") != "complete" or [r["step"] for r in metrics] != list(range(369))
                or [r["tokens"] for r in metrics] != [min(i*262144,96242176) for i in range(369)]
                or [r["step"] for r in metrics if "validation_nll" in r] != plan["evaluation_steps"]
                or summary["steps"] != 368 or summary["tokens"] != 96242176):
            raise ValueError("A run has incomplete horizon or evaluation cadence")
        for record in metrics:
            for key in ("train_nll","gradient_norm_before_clip","validation_nll","train_probe_nll","step_seconds"):
                if key in record and not math.isfinite(record[key]):
                    raise ValueError("Nonfinite completed trajectory")
        saved = torch.load(root/"final_validation.pt",map_location="cpu",weights_only=True)
        if any(saved[k].dtype != torch.long or not torch.equal(saved[k],v) for k,v in pop.items()):
            raise ValueError("Saved evaluation population differs from every declared development target")
        loss = saved["sequence_nll"]
        if loss.shape != pop["target_counts"].shape or not torch.isfinite(loss).all():
            raise ValueError("Invalid saved per-window loss")
        mean = float((loss*pop["target_counts"]).sum()/pop["target_counts"].sum())
        if any(not math.isfinite(value) or abs(mean-value)>1e-12 for value in
               (summary["full_validation_nll"],summary["final_validation_nll"],metrics[-1]["validation_nll"])):
            raise ValueError("Saved all-target losses do not reconstruct the endpoint")
        with torch.serialization.safe_globals([TorchVersion]):
            snapshot = torch.load(root/"final.pt",map_location="cpu",weights_only=True)
        if snapshot["config"] != cfg or snapshot["metadata"] != meta or not all(torch.isfinite(t).all() for t in tensors(snapshot)):
            raise ValueError("Final snapshot identity/finiteness failure")
        if cfg["method"] in ("pd","ts","spd"):
            states = snapshot["optimizer"]["model_statistics"]
            roots = snapshot["optimizer"]["external"]["data_norm"]["roots"]
            if len(states)!=48 or len(roots)!=48 or any(r[1]!=361 for r in roots.values()):
                raise ValueError("Final statistic/root refresh inventory differs")
            if any(int(s["_total_forwards"])!=46994 or int(s["_step_forwards"])!=18
                   or any(abs(float(s[k])-v)>1e-7 for k,v in weights.items()) for s in states.values()):
                raise ValueError("Full/partial microforward statistic clocks differ")
        del snapshot
        sums = torch.zeros(len(documents),dtype=torch.float64).scatter_add_(0,pop["document_indices"],loss.double()*pop["target_counts"])
        counts = torch.zeros(len(documents),dtype=torch.long).scatter_add_(0,pop["document_indices"],pop["target_counts"])
        groups = []
        for group in range(4):
            indices = torch.tensor([i for i,d in enumerate(documents) if int(d["identity"][:2],16)%4==group])
            total = int(counts[indices].sum())
            groups.append(dict(group=group,targets=total,nll=float(sums[indices].sum()/total)))
        row.update(full_development_nll=mean,development_groups=groups,training_seconds=summary["training_seconds"],
                   total_seconds=summary["total_seconds"],evaluations=[r for r in metrics if "validation_nll" in r],
                   peak_cuda_allocated_bytes=summary["peak_cuda_allocated_bytes"],
                   artifact_sha256={name:sha(root/name) for name in ("ARM_EXECUTION.json","config.json","metadata.json","summary.json","metrics.jsonl","final_validation.pt","final.pt")})
        rows.append(row)
    if len({p[0] for p in paired.values()}) != 2 or len({p[1] for p in paired.values()}) != 2:
        raise ValueError("The two fixed seeds do not have distinct initialization/data streams")
    selected = select_rates(rows)
    pairs,passed = ordering_gates(selected)
    requests = {m:s["proposed_single_outward_lr"] for m,s in selected.items()
                if s["proposed_single_outward_lr"] is not None} if stage=="initial" else {}
    result = dict(stage=stage,development_only=True,sealed_test_scored=False,full_goal_qualified=False,
        rows=rows,joint_lr_selections=selected,adjacent_pairs=pairs,ordering_viability_passed=passed,
        boundary_requests=requests,plan_sha256=sha(cohort/"PLAN.json"),source_manifest_sha256=sha(cohort/"source_manifest.json"),
        decision="Run only the declared boundary checks before finalizing ordering." if requests else
                 "Development ordering passes; batch/momentum/independent confirmation remain required." if passed else
                 "The predeclared ordering/bracketing requirement fails; preserve and close this candidate.")
    out.mkdir(exist_ok=False)
    freeze(out/"frozen_readout")
    result["readout_source_manifest_sha256"] = sha(out/"source_manifest.json")
    write(out/"results.json",result)
    shutil.copy2(__file__,out/"ordering_d_report.py")
    lines=["# Candidate D development ordering", "",result["decision"],"", "| Method | Joint selected LR | Mean full NLL | Bracket closed |", "|---|---:|---:|---|"]
    for method,item in selected.items():
        winner=item["selected"]
        lines.append(f"| {method} | {winner['lr'] if winner else 'none'} | {winner['mean_nll'] if winner else 'unstable'} | {item['bracket_closed']} |")
    lines += ["","All model scores are development evidence. Both fixed seeds select one common learning rate per method; no favorable window subset or per-seed LR envelope is used. Every declared adjacent sign, material-gap requirement and bracket is reported in results.json. The complete surrogate goal is not yet qualified."]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    try:
        plot_readout(result,out)
        write(out/"PLOTS.json",{name:sha(out/name) for name in ("ordering_curves.png","ordering_curves.pdf")})
    except Exception as error:
        # Plotting is a presentation step; preserve the validated numerical
        # readout and its immutable artifacts even if rendering is unavailable.
        write(out/"PLOT_FAILURE.json",dict(error=repr(error),numerical_readout_unchanged=True))
    return result


def prepare_edges(cohort):
    """Freeze only boundary checks implied by the complete initial readout."""
    cohort=Path(cohort).resolve()
    plan=verify(cohort,"initial")
    required_configs(cohort,"initial")
    path=cohort/"report_initial/results.json"
    initial=read(path)
    if (initial["stage"]!="initial" or initial["plan_sha256"]!=sha(cohort/"PLAN.json")
            or initial["source_manifest_sha256"]!=sha(cohort/"source_manifest.json")
            or (cohort/"EDGE_PREFLIGHT.json").exists()):
        raise ValueError("Changed initial identity or edge stage already frozen")
    for row in initial["rows"]:
        for name,digest in row["artifact_sha256"].items():
            if sha(cohort/"runs"/row["run_id"]/name)!=digest:
                raise ValueError("Initial run changed after its readout")
    selected=select_rates(initial["rows"])
    requests={m:s["proposed_single_outward_lr"] for m,s in selected.items()
              if s["proposed_single_outward_lr"] is not None}
    if requests!=initial["boundary_requests"]:
        raise ValueError("Boundary requests do not reconstruct")
    if not requests:
        return dict(edges_needed=False,ordering_viability_passed=initial["ordering_viability_passed"])
    configs=required_configs(cohort,"initial")
    tasks,hashes=[],{}
    for method,lr in requests.items():
        for seed in SEEDS:
            cfg=dict(next(c for c in configs if c["method"]==method and c["seed"]==seed))
            cfg.update(lr=lr,run_id=f"{method}_D_lr{lr:g}_m0.9_s{seed}")
            target=cohort/"configs"/(cfg["run_id"]+".json")
            write(target,cfg)
            hashes[target.name]=sha(target)
            tasks.append([dict(config=str(target),out=str(cohort/"runs"/cfg["run_id"]))])
    if len(tasks)>10:
        raise ValueError("Original edge allowance exceeded")
    random.Random(20260928).shuffle(tasks)
    write(cohort/"tasks_edges.json",tasks)
    make_script(cohort,Path(__file__).resolve().parents[2],"edges",plan["worker_process_seconds"])
    write(cohort/"EDGE_PREFLIGHT.json",dict(plan_sha256=sha(cohort/"PLAN.json"),initial_report_sha256=sha(path),
        edges_config_sha256=hashes,tasks_edges_sha256=sha(cohort/"tasks_edges.json"),
        permitted_requests=requests,maximum_outward_rate_per_method=1,automatic_retry=False))
    return dict(edges_needed=True,arms=len(tasks),requests=requests)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",type=Path)
    parser.add_argument("--stage",choices=("initial","final"),default="initial")
    parser.add_argument("--prepare-edges",action="store_true")
    args=parser.parse_args()
    if args.prepare_edges:
        print(json.dumps(prepare_edges(args.cohort),indent=2))
    else:
        value=report(args.cohort,args.stage)
        print(json.dumps({k:value[k] for k in ("ordering_viability_passed","boundary_requests","decision")},indent=2))
