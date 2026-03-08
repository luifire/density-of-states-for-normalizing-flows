source activate training
export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"
export WANDB_AGENT_MAX_INITIAL_FAILURES=1000
wandb agent $1  | tee sweep_log.txt