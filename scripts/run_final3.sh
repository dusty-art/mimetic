#!/bin/zsh
# Full final pipeline (v2: eyelid transfer + upper-face fitting). Sequential: every stage is GPU-bound.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
STEPS=${STEPS:-14000}
log() { echo "$(date +%H:%M:%S) $*" | tee -a experiments/final_progress.log; }
log "v2: motion database"
python scripts/build_motion.py --workers 6 > experiments/final_motion.log 2>&1
log "v2: muscle fitting"
rm -f checkpoints/encoder.pt
python scripts/fit_muscles.py --epochs 6 --calibrate > experiments/final_fit.log 2>&1
log "v2: controller dataset"
python scripts/build_ctrl_dataset.py > experiments/final_ctrl_dataset.log 2>&1
log "v2: ARKit map"
python scripts/train_arkit_map.py > experiments/final_arkit.log 2>&1
log "v2: controller"
python scripts/train_controller.py --steps $STEPS --name controller > experiments/final_controller.log 2>&1
log "v2: audio model"
rm -f data/tts/teacher.npz
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
