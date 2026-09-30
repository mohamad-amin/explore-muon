#!/bin/bash
# Run Track 3 scripts one after another as a Slurm step on one of our allocations (4 GPUs),
# with the system compilers (the conda CC/CXX break torch.compile).
#   track3/run_on.sh JOBID SCRIPT [SCRIPT ...]     JOBID: 2618555 = g20 (RTX 6000 Ada), 2567578 = priv-g14 (L40S)
# Track 3 logs go to track3/logs/<uuid>.txt, console output to track3/logs/console_*.txt.
cd "$(dirname "$0")"
jobid=$1; shift
for script in "$@"; do
  console="logs/console_${script%.py}_${jobid}_$(date -u +%Y%m%dT%H%M%S).txt"
  echo "[$(date -u +%FT%TZ)] start $script on $jobid -> $console"
  srun --jobid="$jobid" --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --chdir="$PWD" \
    bash -c 'exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 \
      /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -m torch.distributed.run \
      --standalone --nproc_per_node=4 '"$script" > "$console" 2>&1
  code=$?
  echo "[$(date -u +%FT%TZ)] end $script on $jobid exit $code"
done
