source activate training

# Adjust this if you want to run the script in another dir
export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"

# Unit Test
#../unit_tests/unit_tests.sh

#old use case used augmentation --eval_batch_size=1024  --eval_every 10

# Main Training
python ../../src/models/master/training/start_training_master.py --model demun \
				--epochs 500 --dataset mnist --batch_size 32 \
				--lr 1e-4 --sample_every 5 --eval_every 1 \
				--root /master-thesis-work/data/ --training_class 0


echo 'Done Main Training'
