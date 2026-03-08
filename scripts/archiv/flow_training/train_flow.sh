

# remove old saving stuff
rm -r /master-thesis-work/data/checkpoints

# this is just here to capture the output
./execute.sh | tee log_training_nf.txt

./upload.sh

# cause curl forgets the line break
echo 
echo "Done with the Flow"