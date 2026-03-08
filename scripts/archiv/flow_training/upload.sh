
rm nf_mnist.zip
zip -r nf_mnist.zip /master-thesis-work/data/checkpoints log_training_nf.txt
curl --upload-file nf_mnist.zip https://oshi.at | tee download_nf_mnist.txt
# alternativ transfer.sh