source activate training

# Adjust this if you want to run the script in another dir
export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"

# Unit Test
../unit_tests/unit_tests.sh

#old use case used augmentation, eval_batch_size, eval_every 10

# Main Training
python ../../src/models/master/training/start_training_master.py --model survae \
				--epochs 500 --dataset mnist --batch_size 32 --sample_every 5 --eval_every 1 \
				--optimizer adamax --lr 1e-3 --gamma 0.995  --warmup 5000 \
				--num_steps 12 --num_scales 2 --dequant flow --pooling max \
				--root /master-thesis-work/data/ --num_workers 5 \
				--training_class 0

echo 'Done Main Training'

#echo 'usually use 500 epochs'
# Cooldown Training
#python ../../models/survae_flows/experiments/image/train_more_master.py --new_epochs 550 --new_lr 2e-5 \
#			--model /master-thesis-work/models/survae_flows/experiments/image/log/mnist_8bit/pool_flow/expdecay/survae_mnist_master/

echo 'Done Cooldown'
