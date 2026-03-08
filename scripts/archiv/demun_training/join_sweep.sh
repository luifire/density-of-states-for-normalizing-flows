source activate training
export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"
wandb agent $1  | tee sweep_log.txt