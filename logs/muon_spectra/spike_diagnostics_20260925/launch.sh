#!/bin/bash
# Spike-origin diagnostics (MUON_CASE.md, 2026-09-25): one L40S GPU per final
# checkpoint inside allocation 2567578 (priv-g14). Usage: launch.sh gates|probe|science [tag]
set -u
cd /share/data/dl-theory/amin/projects/explore_muon
ROOT=logs/muon_spectra/spike_diagnostics_20260925
MODE=$1
TAG=${2:-}
case $MODE in
  gates)   EXTRA="--gate-only" ;;
  probe)   EXTRA="--batches 2" ;;
  science) EXTRA="--batches 64" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
declare -A RUNS=(
  [muon_d8]=logs/muon_spectra/depth8_w512_20260925_r2/scientific
  [muon_d12]=logs/muon_spectra/depth12_w512_20260925_r2/scientific
  [adamw_d8]=logs/adamw_spectra/g20_20260924_212246_r2/77m_seed260924
  [adamw_d12]=logs/adamw_spectra/depth12_w512_20260924/scientific
)
OUT=$ROOT/$MODE$TAG
test ! -e $OUT || { echo "refusing to reuse $OUT"; exit 2; }
mkdir -p $OUT
i=0
for name in muon_d8 muon_d12 adamw_d8 adamw_d12; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 .venv/bin/python -u -m research.adamw_spectra.spike_diagnostics \
    --run ${RUNS[$name]} --out $OUT/$name $EXTRA > $OUT/$name.log 2>&1 &
  pids[$i]=$!
  i=$((i+1))
done
status=0
for pid in ${pids[@]}; do wait $pid || status=1; done
echo "mode=$MODE$TAG exit=$status"
exit $status
