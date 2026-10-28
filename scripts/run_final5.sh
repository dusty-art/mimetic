#!/bin/zsh
# Evaluation onward (after the evaluate.py batch-window fix).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "$(date +%H:%M:%S) $*" | tee -a experiments/final_progress.log; }
log "v2: evaluation (rerun)"
python scripts/evaluate.py > experiments/final_eval.log 2>&1
log "v2: composition"
python scripts/composition_test.py > experiments/final_composition.log 2>&1
log "v2: emotion cues"
python scripts/emotion_cues.py > experiments/final_emotion_cues.log 2>&1
log "v2: figures"
python scripts/figures.py > experiments/final_figures.log 2>&1
log "v2: done"
