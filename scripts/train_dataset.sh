#!/bin/bash

export PATH="~/miniconda3/bin:$PATH"

source activate training

# Adjust this if you want to run the script in another dir
export PYTHONPATH="${PYTHONPATH}:/home/luibrand/master-thesis-work/src/models:/home/luibrand/master-thesis-work/src/common:/home/luibrand/master-thesis-work/src/exp"

# Main Training
python ../src/models/master/train_settings/multi_training_normal.py

echo 'Done Main Training'
