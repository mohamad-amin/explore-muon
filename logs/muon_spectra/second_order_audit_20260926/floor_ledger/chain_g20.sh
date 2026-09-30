#!/bin/bash
# After the @46 harness pair: gentle exits (plain PD at 0.1x LR for 4 steps from the staged and control @46 states),
# the staged-correction noise probe at the staged @46 state, then band-restricted coordination (--stage-band top) from
# 16M @9 to 92 (MUON_CASE 2026-09-29 07:37 / 07:38 CDT).
cd /share/data/dl-theory/amin/projects/explore_muon
A=logs/muon_spectra/second_order_audit_20260926
RUN=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s
TR=".venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4"
until [ -f $A/floor_ledger/staged_46/done.json ] && [ -f $A/floor_ledger/control_46/done.json ]; do sleep 20; done
for src in staged control; do
  out=$A/floor_ledger/exit_${src}_0.1
  mkdir -p $out && cp $A/floor_ledger/${src}_46/kept/step000046.pt $out/checkpoint.pt
  $TR $A/newton_train.py $out $RUN:9 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-off --lr-scale 0.1 --validation-every 1 --sharpness-every 2 --stop-after 41 --checkpoint-every 1000 > $out.log 2>&1
  rm -f $out/checkpoint.pt
done
$TR $A/staged_noise_probe.py $A/floor_ledger/noise_staged_46.json $RUN:9 --harness $A/floor_ledger/staged_46/kept/step000046.pt --sizes 32,256 > $A/floor_ledger/noise_staged_46.log 2>&1
$TR $A/newton_train.py $A/band/band_top $RUN:9 --normalize 0 --momentum 0.9 --clip 1.0 --staged-pd 0.5 --stage-band top --validation-every 10 --sharpness-every 10 --checkpoint-every 1000 > $A/band/band_top.log 2>&1
