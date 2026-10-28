#!/bin/zsh
# Final training pipeline, run after tracking and alignment have finished.
# Each stage logs to experiments/final_<stage>.log; the script stops at the first failure.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
STEPS=${STEPS:-14000}
log() { echo "$(date +%H:%M:%S) $*" | tee -a experiments/final_progress.log; }

log "waiting for tracking + alignment"
until grep -q "finished" data/processed/_log.txt 2>/dev/null; do sleep 30; done
until [ "$(find data/processed -name '*.npz' ! -name '*.tmp.npz' | wc -l)" -le "$(find data/processed -name '*.phn.json' | wc -l)" ]; do sleep 30; done

log "motion database"
python scripts/build_motion.py --workers 6 > experiments/final_motion.log 2>&1
log "muscle fitting"
rm -f checkpoints/encoder.pt
python scripts/fit_muscles.py --epochs 6 --calibrate > experiments/final_fit.log 2>&1
log "controller dataset"
python scripts/build_ctrl_dataset.py > experiments/final_ctrl_dataset.log 2>&1
log "ARKit map"
python scripts/train_arkit_map.py > experiments/final_arkit.log 2>&1
log "controller"
python scripts/train_controller.py --steps $STEPS --name controller > experiments/final_controller.log 2>&1
log "baseline: direct regression"
python scripts/train_controller.py --steps $STEPS --baseline direct --name baseline_direct > experiments/final_baseline_direct.log 2>&1
log "ablation: no articulation constraints"
python scripts/train_controller.py --steps $STEPS --no-artic --name controller_noartic > experiments/final_noartic.log 2>&1
log "audio model"
python scripts/train_audio_model.py --steps ${AUDIO_STEPS:-16000} > experiments/final_audio.log 2>&1
log "evaluation"
python scripts/evaluate.py > experiments/final_eval.log 2>&1
log "done"
