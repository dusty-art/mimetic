#!/bin/zsh
# Second half of the final pipeline: waits for the three controllers, then audio model, evaluation, figures.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "$(date +%H:%M:%S) $*" | tee -a experiments/final_progress.log; }
until grep -q "^saved" experiments/final_controller.log; do
  if grep -q "Traceback" experiments/final_controller.log; then log "FAILED: controller"; exit 1; fi
  sleep 30
done
# The audio model needs the controller (TTS teacher); comparisons run after it, one at a time (GPU-bound).
log "audio model"
python scripts/train_audio_model.py --steps ${AUDIO_STEPS:-16000} > experiments/final_audio.log 2>&1
log "baseline: direct regression"
python scripts/train_controller.py --steps 14000 --baseline direct --name baseline_direct > experiments/final_baseline_direct.log 2>&1
log "ablation: no articulation constraints"
python scripts/train_controller.py --steps 14000 --no-artic --name controller_noartic > experiments/final_noartic.log 2>&1
log "evaluation"
python scripts/evaluate.py > experiments/final_eval.log 2>&1
log "composition test"
python scripts/composition_test.py > experiments/final_composition.log 2>&1
log "figures"
python scripts/figures.py > experiments/final_figures.log 2>&1
log "done"
