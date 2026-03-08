

rm survae_mnist.zip
zip -r survae_mnist.zip survae_mnist_master.txt /master-thesis-work/data/checkpoints
#zip -r survae_mnist.zip /master-thesis-work/checkpoints/model/survae_steps-12_scales-2_pool-max_mnist_cls-0/2021-04-20_11-53-10/samples/
curl --upload-file survae_mnist.zip https://oshi.at | tee download_link_survae_mnist.txt
# alternativ transfer.sh

