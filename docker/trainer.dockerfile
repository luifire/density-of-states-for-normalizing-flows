#FROM ubuntu:20.04
#from nvcr.io/nvidia/pytorch
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04
#CMD nvidia-smi


############################################################################
# Utils

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get upgrade -y
#RUN DEBIAN_FRONTEND="noninteractive"
#RUN apt-get install -y ssh 
RUN apt-get install -y nano
RUN apt-get install -y zip unzip 
RUN apt-get install -y curl

############################################################################
# Install Conda

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version


############################################################################
# Setup Conda Env

COPY ./env.yml .
RUN conda env create -f env.yml --name training python=3.8

# add this to start building below 
# ARG CACHEBUST=1

############################################################################
# copy all datasets into docker file
# Warning: Datasets need to be in the same folder as the docker.file
ADD datasets/ /datasets

############################################################################
# GIT Repo

RUN apt-get update && apt-get upgrade -y
RUN apt-get install git -y

# add this and below command will run without cache
RUN git clone https://luifire:yvDyDzZy5rga9RHkbQ9V@git.tu-berlin.de/luifire/master-thesis-work.git 

############################################################################
# this makes you use bash instead of shell, which allows you to execute the script below properly
SHELL ["/bin/bash", "-c"]

# sets the logging key
RUN /master-thesis-work/docker/set_up_wandb.sh

