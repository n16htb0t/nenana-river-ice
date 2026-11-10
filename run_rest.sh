#!/usr/bin/env bash
# Second batch, after run_all.sh: independent one-vs-rest models, the four-class
# UNet++ rerun that saves its cross-validated predictions (oof.npz), predictions
# on the unlabelled dates for every task, river/land state masks, and images.
# Usage: bash ~/Desktop/nenana_crops/daksh/run_rest.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=~/venvs/river-ml/bin/python
export PYTHONUNBUFFERED=1

while pgrep -f "run_all.sh" > /dev/null; do sleep 60; done

for ds in exclude_daksh with_daksh; do
    for task in ind_river ind_ice ind_snow ind_land; do
        echo "=== $ds / $task / Random Forest"
        $PY train_rf.py --task $task --dataset $ds
        echo "=== $ds / $task / UNet++"
        $PY train_unetpp.py --task $task --dataset $ds
        $PY evaluate.py --task $task --dataset $ds
    done
done

# the first four-class UNet++ run predates oof.npz; rerun so its block scores exist
if [ ! -f output/1_exclude_daksh/four_class_river_ice_snow_land/unet_plus_plus/oof.npz ]; then
    echo "=== exclude_daksh / four / UNet++ (rerun for oof.npz)"
    $PY train_unetpp.py --task four --dataset exclude_daksh
    $PY evaluate.py --task four --dataset exclude_daksh
fi

for ds in exclude_daksh with_daksh; do
    echo "=== $ds / unlabelled dates"
    for task in four snow river ind_river ind_ice ind_snow ind_land; do
        $PY predict_new.py --task $task --dataset $ds
    done
    $PY predict_new.py --task combined --dataset $ds
    echo "=== $ds / river-land state masks"
    $PY scene_mask.py --dataset $ds
    echo "=== $ds / images"
    $PY images.py --dataset $ds
done
echo "REST DONE"
