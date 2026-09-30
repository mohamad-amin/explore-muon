#!/bin/bash
# Submit a Track 3 script as a gpu-partition job on 4 GPUs of one type (default L40S), system compilers.
#   [CPUS=8] [TIME=04:00:00] track3/submit_l40s.sh SCRIPT [GRES]      e.g. GRES=gpu:nvidia_rtx_a6000:4
cd "$(dirname "$0")"
script=$1; gres=${2:-gpu:nvidia_l40s:4}; name=${script%.py}
sbatch --job-name="t3-${name#train_gpt_}" --partition=gpu --nodes=1 --ntasks=1 --gres="$gres" \
  --cpus-per-task=${CPUS:-16} --mem=96G --time=${TIME:-03:00:00} --chdir="$PWD" \
  --output="$PWD/logs/console_${name}_%j.txt" --error="$PWD/logs/console_${name}_%j.txt" \
  --wrap='exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 '"$script"
