FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        bzip2 \
        unzip \
        make \
        wget \
        git \
        python3 \
        python3-dev \
        python3-websockets \
        python3-setuptools \
        python3-pip \
        python3-wheel \
        zlib1g-dev \
        patch \
        ca-certificates \
        swig \
        cmake \
        xz-utils \
        automake \
        autoconf \
        libtool \
        pkg-config 

COPY . /app/server/.

WORKDIR /app/server

RUN pip3 install -r requirements.txt

CMD python3 app.py