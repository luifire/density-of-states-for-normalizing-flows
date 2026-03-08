source activate training
export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"
wandb sweep --project sweeps --name $1 $2/sweep.yaml
