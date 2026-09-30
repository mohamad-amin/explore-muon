#!/bin/bash
# Cooldown test on g20 (4 Ada GPUs) once the 16M queue there has finished; cancels the pending gpu-partition copies.
D=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926; SRC=/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s:1300
until python3 -c "import json,sys; d=json.load(open('/share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/queue_batch16m_g20.json')); sys.exit(0 if 'ended_unix' in d else 1)" 2>/dev/null; do sleep 30; done
for j in 2625350 2625351 2625353; do [ "$(squeue -h -j $j -o %T 2>/dev/null)" = "PENDING" ] && scancel $j && echo "cancelled pending $j"; done
date "+gate passed %H:%M"
run() { gpus=$1; out=$2; shift 2; srun --jobid=2618555 --overlap -N1 -n1 --cpus-per-task=8 bash -c "export CUDA_VISIBLE_DEVICES=$gpus OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1; cd /share/data/dl-theory/amin/projects/explore_muon; exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 $D/newton_train.py $D/cooldown_gnpd/$out $SRC --momentum 0.95 --validation-every 25 --checkpoint-every 25 $*" >> $D/cooldown_gnpd/$out.log 2>&1; echo "$out exit $? $(date +%H:%M)"; }
( run 2,3 B3_gnpd_p0.25_lr3x --normalize 0.001 --gnpd 0.25 --cg 64 --lr-scale 3 --stop-after 169 ) &
run 0,1 A_polar --normalize 0 --polar-control --stop-after 150
cp -r $D/cooldown_gnpd/A_polar $D/cooldown_gnpd/C_polar_then_gnpd_lr3x && echo "copied A@1450 for C"
run 0,1 A_polar --normalize 0 --polar-control --stop-after 169
run 0,1 C_polar_then_gnpd_lr3x --normalize 0.001 --gnpd 0.25 --cg 64 --lr-scale 3 --stop-after 169
run 0,1 B_gnpd_p0.25 --normalize 0.001 --gnpd 0.25 --cg 64 --stop-after 169
wait
date "+all done %H:%M"
