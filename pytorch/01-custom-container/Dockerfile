FROM nvidia/cuda:9.0-runtime

RUN apt-get update && \
    apt-get -y install build-essential python-dev python3-dev python3-pip python-imaging wget curl

COPY mnist_cnn.py /opt/program/train
RUN chmod +x /opt/program/train

RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl --upgrade && \
    pip3 install torchvision --upgrade
    
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program