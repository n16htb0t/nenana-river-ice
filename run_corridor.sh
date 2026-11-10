#!/usr/bin/env bash
# Four-class RF + UNet++ inside the river corridor only (river + 150 m of bank each side),
# for exclude_daksh, with_daksh and the external test; then maps of the unlabelled dates,
# river/land state masks, figures, JPEGs and summaries. Outputs: output/4_river_corridor_150m/
# Usage: bash ~/Desktop/nenana_crops/daksh/run_corridor.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=~/venvs/river-ml/bin/python
export PYTHONUNBUFFERED=1
export RIVER_CORRIDOR_M=150

for ds in exclude_daksh with_daksh; do
    echo "=== corridor / $ds / Random Forest"
    $PY train_rf.py --task four --dataset $ds
    echo "=== corridor / $ds / UNet++"
    $PY train_unetpp.py --task four --dataset $ds
    $PY evaluate.py --task four --dataset $ds
    $PY predict_new.py --task four --dataset $ds
    $PY scene_mask.py --dataset $ds
    $PY images.py --dataset $ds
done
echo "=== corridor / external test"
$PY external_test.py --task four
$PY evaluate.py --task four --dataset external_daksh
$PY scene_mask.py --dataset external_daksh
$PY images.py --dataset external_daksh
$PY summarise.py
echo "CORRIDOR DONE"
