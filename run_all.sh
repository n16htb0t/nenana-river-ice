#!/usr/bin/env bash
# Every model, task and dataset, then combine + evaluate. Outputs land in
#   output/<exclude_daksh | with_daksh>/four_class/{rf,unetpp,evaluation}
#   output/<exclude_daksh | with_daksh>/binary/{snow,river,combined}/{rf,unetpp,evaluation}
# Usage: bash ~/Desktop/nenana_crops/daksh/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=~/venvs/river-ml/bin/python
export PYTHONUNBUFFERED=1

# the first four-class UNet++ run (exclude_daksh) wrote to output/{rf,unetpp}
while pgrep -f "train_unetpp.py$" > /dev/null; do sleep 30; done
if [ -d output/unetpp ]; then
    mkdir -p output/exclude_daksh/four_class
    rm -rf output/exclude_daksh/four_class/unetpp
    mv output/unetpp output/exclude_daksh/four_class/unetpp
    rm -rf output/rf output/evaluation
fi

for ds in exclude_daksh with_daksh; do
    for task in four snow river; do
        echo "=== $ds / $task / Random Forest"
        $PY train_rf.py --task $task --dataset $ds
        if [ "$ds/$task" = "exclude_daksh/four" ] && [ -f output/exclude_daksh/four_class/unetpp/metrics.txt ]; then
            echo "=== $ds / $task / UNet++ already done"
        else
            echo "=== $ds / $task / UNet++"
            $PY train_unetpp.py --task $task --dataset $ds
        fi
        $PY evaluate.py --task $task --dataset $ds
    done
    echo "=== $ds / binary combined"
    $PY combine.py --dataset $ds
    $PY evaluate.py --task combined --dataset $ds
done
echo "ALL DONE"
