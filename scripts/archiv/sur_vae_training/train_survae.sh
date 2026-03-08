

# remove old saving stuff
rm -r /master-thesis-work/data/checkpoints

# this is just here to capture the output
./execute.sh | tee survae_mnist_master.txt


#curl --upload-file survae_mnist.zip https://oshi.at | tee download_link_survae_mnist.txt
./upload.sh

# cause curl forgets the line break
echo 
echo "Done with the Flow"