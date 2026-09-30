#!/bin/bash
# Start the priv-g14 16M queue once the normalized-GN run (2 of the node's 4 GPUs) has finished.
until [ -f /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/second_order_audit_20260926/newton_ref/M1M_normgn_k64_d1e-3_from500/done.json ]; do sleep 20; done
date "+priv-g14 gate passed %H:%M" > /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/launcher_privg14.log
exec srun --jobid=2567578 --overlap -N1 -n1 --cpus-per-task=16 bash -c "export CUDA_VISIBLE_DEVICES=0,1,2,3; exec /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/start_privg14.sh" >> /share/data/dl-theory/amin/projects/explore_muon/logs/muon_spectra/soaudit_batch16m_20260927/launcher_privg14.log 2>&1
