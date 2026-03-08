source activate training

# Adjust this if you want to run the script in another dir
export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"

# Unit Test
../unit_tests/unit_tests.sh

#old use case used augmentation, eval_batch_size, eval_batch_size=4096

# Main Training
python ../../src/models/master/training/start_training_master.py --model nf \
			--flow=RealNVP --dataset=mnist --epochs=500 --sample_every 5 --eval_every 10 \
			--lr=5e-5 \
			--prior=Gaussian --num_blocks=6 --batch_size=32 \
			--num_workers=0 --root /master-thesis-work/data/ \
			--training_class 0 --old_use_case --wandb_name_ext #

echo 'Done Main Training'


