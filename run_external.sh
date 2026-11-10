#!/usr/bin/env bash
# Third batch, after run_rest.sh: external test cohort (train on the patch dates,
# score on the labelled daksh date) for every task, then combine, state masks, images.
# Usage: bash ~/Desktop/nenana_crops/daksh/run_external.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=~/venvs/river-ml/bin/python
export PYTHONUNBUFFERED=1

while pgrep -f "run_all.sh|run_rest.sh" > /dev/null; do sleep 60; done

for task in four snow river ind_river ind_ice ind_snow ind_land; do
    echo "=== external_daksh / $task"
    $PY external_test.py --task $task
    $PY evaluate.py --task $task --dataset external_daksh
done
$PY combine.py --dataset external_daksh
$PY evaluate.py --task combined --dataset external_daksh
$PY scene_mask.py --dataset external_daksh
$PY images.py --dataset external_daksh

$PY summarise.py
echo "EXTERNAL DONE"
