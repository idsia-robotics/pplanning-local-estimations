FROM ubuntu:bionic

MAINTAINER Jerome Guzzi "jerome@idsia.ch"

RUN apt-get -y update && apt-get install -y \
    lsb-core \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN wget http://ompl.kavrakilab.org/install-ompl-ubuntu.sh && \
    chmod +x install-ompl-ubuntu.sh && \
    ./install-ompl-ubuntu.sh --python

RUN pip3 install jupyter matplotlib

RUN pip3 install keras tensorflow scikit-image h5py tqdm

RUN pip3 install PyYAML
