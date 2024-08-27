# Define base image/operating system
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates pkg-config libssl-dev libopenblas-dev
# RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda3 -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh

# Copy files and directory structure to working directory
COPY . . 
#COPY bashrc ~/.bashrc
ENV PATH=/miniconda3/bin:~/.cargo/bin/:${PATH}

SHELL ["/bin/bash", "--login", "-c"]
RUN conda init bash
#RUN echo 'export PATH=/miniconda3/bin:$PATH' > ~/.bashrc

RUN conda create -n hiob  python=3.11
RUN conda run -n hiob pip install numpy
RUN conda run -n hiob pip install h5py
RUN conda run -n hiob pip install maturin
RUN conda run -n hiob pip install matplotlib
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rust-installer.sh
RUN conda run -n hiob bash rust-installer.sh -y
RUN rm rust-installer.sh
ENV RUSTFLAGS=" -C target-cpu=native -C opt-level=3"
RUN cd rust-search && conda run -n hiob cargo build -r
# the following avoids some rare error of bash
# RUN cp /lib/x86_64-linux-gnu/libtinfo.so.6 /miniconda3/envs/hiob/lib/libtinfo.so.6

# Set container's working directory - this is arbitrary but needs to be "the same" as the one you later use to transfer files out of the docker image
#WORKDIR result


# Run commands specified in "run.sh" to get started

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
