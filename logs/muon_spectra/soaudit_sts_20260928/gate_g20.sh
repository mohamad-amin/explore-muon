#!/bin/bash
# After both GN-PD floor-step arms have logged step 4 (one-time-release measurement), stop them and start the 16M S∘TS arm.
until grep -q '"k": 4,' /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/floor_steps/a516_gnpd_k1.log && grep -q '"k": 4,' /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/floor_steps/a516_gnpd_k2.log; do sleep 30; done
pkill -f "[f]loor_steps.py.*--direction gnpd" && echo "stopped GN-PD floor arms $(date +%H:%M)"
sleep 20
date "+g20 start %H:%M"
exec srun --jobid=2618555 --overlap -N1 -n1 --cpus-per-task=16 bash -c "export CUDA_VISIBLE_DEVICES=0,1,2,3; exec /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_sts_20260928/start_g20.sh"
