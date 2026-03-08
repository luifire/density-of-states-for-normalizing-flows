#!/bin/bash

export PATH="~/miniconda3/bin:$PATH"

source activate training

wandb offline

# Adjust this if you want to run the script in another dir
export PYTHONPATH="${PYTHONPATH}:/home/luibrand/master-thesis-work/src/models:/home/luibrand/master-thesis-work/src/common:/home/luibrand/master-thesis-work/src/exp"

# Main Training
python ../src/models/master/training/start_training_master.py  --root=/home/luibrand/master-thesis-work/data/ \
		--model=nf --flow=RealNVP --dataset=mnist --epochs=100 --eval_every=10 \
		--dimensions=784 --batch_size=32 --st_type=resnet \
		--sample_every=10 --inflation --st_type=resnet --early_stopping=50 --eval_batch_size=1024 \
		--wandb_name_ext=cluster --lr=1e-4 --num_blocks=8 --dont_check --distance_evaluation=10 --var=0.0001

# var or uniform_noise

# Kirichenko learns noise of strength 10^-6 
echo 'Done Main Training'
