FROM ubuntu:16.04

RUN apt-get update && \
    apt-get -y install build-essential libopencv-dev libopenblas-dev libjemalloc-dev libgfortran3 \
    python-dev python3-dev python3-pip wget curl

COPY mnist_cnn.py /opt/program/train
RUN chmod +x /opt/program/train

RUN mkdir /root/.keras
COPY keras.json /root/.keras/

RUN pip3 install mxnet --upgrade --pre && \
    pip3 install keras-mxnet --upgrade --pre

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program