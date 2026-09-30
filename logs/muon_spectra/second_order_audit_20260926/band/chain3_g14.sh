#!/bin/bash
# After the 4M band pair on priv-g14 (MUON_CASE 2026-09-29 10:26 / 10:3x CDT): the correction on each matrix's input-mean
# direction only; the full correction with dual-norm scaling (normalized Gauss-Seidel); the 8 largest input eigenvectors.
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
COMMON="--normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --validation-every 10 --sharpness-every 10 --checkpoint-every 1000"
until [ -f $A/band/band_top_4m/done.json ]; do sleep 20; done
$TR $A/newton_train.py $A/band/band_mean /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 $COMMON --stage-band mean > $A/band/band_mean.log 2>&1
$TR $A/newton_train.py $A/band/staged_dual /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 $COMMON --stage-dual > $A/band/staged_dual.log 2>&1
$TR $A/newton_train.py $A/band/band_top8 /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s:9 $COMMON --stage-band top8 > $A/band/band_top8.log 2>&1
