#!/bin/bash
# Run Track 3 scripts one after another on the g20 allocation (4 GPUs) with the system compilers.
#   track3/run_queue.sh SCRIPT [SCRIPT ...]      (Track 3 logs go to track3/logs/<uuid>.txt)
cd "$(dirname "$0")"
for script in "$@"; do
  console="logs/console_${script%.py}_$(date -u +%Y%m%dT%H%M%S).txt"
  echo "[$(date -u +%FT%TZ)] start $script -> $console"
  srun --jobid=2618555 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --chdir="$PWD" \
    bash -c 'exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 \
      /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -m torch.distributed.run \
      --standalone --nproc_per_node=4 '"$script" > "$console" 2>&1
  echo "[$(date -u +%FT%TZ)] end $script exit $?"
done
