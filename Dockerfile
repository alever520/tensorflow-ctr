FROM ubuntu:16.04

MAINTAINER zhubaowen <zhubaowen@cmcm.com>

RUN apt-get update && apt-get install -y python-pip
RUN apt-get install -y vim
RUN apt-get install -y curl
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip install awscli --upgrade
RUN pip install "tensorflow==1.8.0"




