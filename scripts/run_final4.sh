#!/bin/zsh
# Remaining final stages after the controller (audio model onward).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
STEPS=${STEPS:-14000}
log() { echo "$(date +%H:%M:%S) $*" | tee -a experiments/final_progress.log; }
log "v2: audio model (restart)"
python scripts/train_audio_model.py --steps ${AUDIO_STEPS:-16000} > experiments/final_audio.log 2>&1
log "v2: baseline direct"
python scripts/train_controller.py --steps $STEPS --baseline direct --name baseline_direct > experiments/final_baseline_direct.log 2>&1
log "v2: ablation no-artic"
python scripts/train_controller.py --steps $STEPS --no-artic --name controller_noartic > experiments/final_noartic.log 2>&1
log "v2: evaluation"
python scripts/evaluate.py > experiments/final_eval.log 2>&1
log "v2: composition"
python scripts/composition_test.py > experiments/final_composition.log 2>&1
log "v2: emotion cues"
python scripts/emotion_cues.py > experiments/final_emotion_cues.log 2>&1
log "v2: figures"
python scripts/figures.py > experiments/final_figures.log 2>&1
log "v2: done"
