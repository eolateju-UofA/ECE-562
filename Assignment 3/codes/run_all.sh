#!/usr/bin/env bash
# Runs every configuration for Q5 and Q6 (3 seeds each) in parallel.
#   JOBS=8 bash run_all.sh
# Q5: k in {0, 2, 4} with the clip loss.  Q6: k = 2 with naive / ratio
# (the k = 2 clip run is shared with Q5).  CSVs go to runs/.
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-$(nproc)}
mkdir -p runs logs

{
  for seed in 1 2 3; do
    for k in 0 2 4; do echo "$k clip $seed"; done
    for loss in naive ratio; do echo "2 $loss $seed"; done
  done
} | xargs -P "$JOBS" -L 1 bash -c \
  'python train.py --staleness "$0" --loss "$1" --seed "$2" --log-dir runs \
     > "logs/k$0_$1__seed$2.log" 2>&1 && echo "done k=$0 loss=$1 seed=$2"'
