FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

LABEL maintainer="tong@smail.nju.edu.cn"

RUN apt update
RUN apt install -y git zip
RUN apt install -y cifs-utils nfs-common nfs-kernel-server netbase
# WORKDIR /tmp
# RUN git clone https://github.com/xyq7/InvCompress.git
COPY InvCompress-main.zip .
RUN unzip InvCompress-main.zip

WORKDIR InvCompress-main/codes/

RUN pip install -U pip && pip install -e . && pip install tensorboard

WORKDIR /workspace