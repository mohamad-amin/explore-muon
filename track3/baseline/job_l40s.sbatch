#!/bin/bash
#SBATCH --job-name=track3-36-baseline-l40s
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/share/data/dl-theory/amin/projects/explore_muon/track3/logs/console_baseline_l40s_%j.txt
#SBATCH --error=/share/data/dl-theory/amin/projects/explore_muon/track3/logs/console_baseline_l40s_%j.txt
#SBATCH --chdir=/share/data/dl-theory/amin/projects/explore_muon/track3
# Track 3 result #36, unchanged (upstream copy), on 4xL40S; system compilers (conda CC/CXX break torch.compile).
exec env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CC=/usr/bin/gcc CXX=/usr/bin/g++ OMP_NUM_THREADS=4 \
  /share/data/dl-theory/amin/projects/explore_muon/.venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 baseline/train_gpt_simple.py
