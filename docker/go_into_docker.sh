#echo cd C:/Mega/trunk/master-thesis/others/flows_ood/
#echo expecting execution path to be root


#docker container stop trainer_luibrand
#docker container rm trainer_luibrand
#docker run --gpus all nvidia/cuda:11.2.2-runtime-ubuntu20.04 nvidia-smi
docker container run --gpus all -it -d -p 80:80 --name trainer_luibrand trainer_luibrand

#docker container run --gpus '"device=2"' -it -d -p 80:80 --name trainer_luibrand trainer_luibrand
docker container run --gpus '"device=2"' -it -d  --name trainer_luibrand trainer_luibrand

#docker container run -ti --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 nvidia/cuda


docker exec -w /master-thesis-work/ trainer_luibrand git pull 

echo -----------------------------
echo type the following:
echo ./train_flow_mnist.sh
echo -----------------------------
docker exec -w /master-thesis-work/scripts -it trainer_luibrand bash 
