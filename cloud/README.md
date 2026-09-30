# Google Cloud runs

Project `ucsd-hdsi-tianhaowang` (shared lab project; you are an Editor). gcloud CLI:
`/share/data/dl-theory/amin/tools/google-cloud-sdk/bin/gcloud` (defaults: that project, region `us-central1`).

## Image `explore-muon-v2` (family `explore-muon`)

- Ubuntu 24.04 Deep Learning VM base with NVIDIA driver 580.
- Python 3.11.16 venv at `/opt/muon/.venv` with the exact package lock of the TTIC `.venv`
  (`image/requirements.lock`; torch 2.11.0+cu130). TTIC has Python 3.11.2.
- FineWeb10B: 32 train shards + 1 validation shard from `kjj0/fineweb10B-gpt2`, sha256-verified
  against the TTIC copy (`image/fineweb10B.sha256`).
- TTIC paths are mirrored: `/share/data/dl-theory/amin/projects/{explore_muon,last_layer}` with the same
  `.venv` and `data/fineweb10B` symlinks and the TTIC shard mtimes (`image/fineweb10B.mtimes`). Data
  manifests therefore equal the TTIC ones, and the existing frozen controllers run unchanged.
- Recipe: `image/build_image.sh` (v1), then `image/mirror_ttic_paths.sh` (v2).

## Quota (checked 2026-09-25)

Spot only: H100, H100 Mega, H200 and B200, 128 GPUs per region each. On demand: A100 40 GB (16 per
region) and L4 (32). Spot capacity stocks out often; if a zone fails, try another
(`us-central1-a/b/c`, `us-east4-a/b/c`, `us-west1-a/b`, ...).

## Validation and cost

`logs/muon_spectra/gcp_validation_20260925`: an unchanged wave-9 arm (tuned Muon, seed 260924) on
Spot 4×H100 passed all controller gates and finished at 3.70386, against 3.70689 (RTX 6000 Ada) and
3.70427 (L40S). A width-512, 1469-step arm takes ~13 minutes end to end (0.34 s per step).

Spot list prices in us-east4 (2026-09-25): H100 $5.88 per GPU-hour, so a 4×H100 VM is ~$26/h with
its CPUs and memory; B200 $4.08 per GPU-hour including the host; H200 $5.57. One width-512 arm on
4×H100 costs about $6-7.

## Usage

```bash
cloud/gcp.sh up muon-h100-1 a3-highgpu-4g us-east4-b        # Spot 4xH100
cloud/gcp.sh push muon-h100-1 logs/muon_spectra/<cohort> \
  logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/muon/scientific/metadata.json \
  logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/config_muon.json   # run_arm.py reference files
cloud/gcp.sh ssh muon-h100-1        # then: cd /share/data/dl-theory/amin/projects/explore_muon; run in tmux
cloud/gcp.sh pull muon-h100-1 logs/muon_spectra/<cohort>/<arm>
cloud/gcp.sh down muon-h100-1       # delete when done
```

- Preemption or `MAX_HOURS` (default 12) stops the VM and keeps its disk. A stopped VM still bills
  its disk, so run `down` once results are pulled.
- Compare arms only on the same GPU type, as on TTIC.
- Every resource is labeled `owner=mohamadamin,study=explore-muon` so spend can be separated from the
  other users of the project.
