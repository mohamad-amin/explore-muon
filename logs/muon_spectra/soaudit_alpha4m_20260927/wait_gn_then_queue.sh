#!/bin/bash
# Inside an srun step holding the node's 4 GPUs: wait until the second-order audit's one_step_gn.py measurements on
# this node have finished, then run the arms one after another with node_queue.py.
# usage: wait_gn_then_queue.sh QUEUE_NAME WAIT_FOR_ARM_DIR ARM [ARM ...]
set -u
cd /share/data/dl-theory/amin/projects/explore_muon
while pgrep -f "second_order_audit_20260926/one_step_gn.py" > /dev/null; do sleep 30; done
exec .venv/bin/python -u logs/muon_spectra/soaudit_alpha4m_20260927/node_queue.py "$@"
