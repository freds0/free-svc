# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

FROM nvcr.io/nvidia/pytorch:23.12-py3
RUN apt update && \
    apt -y install git libsndfile1-dev ffmpeg

# RUN python3 -m pip install --upgrade pip

# RUN python3 -m pip install torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu116

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Install fairseq (not necessary now)
# RUN git clone https://github.com/facebookresearch/fairseq.git && \
#     cd fairseq && \
#     git checkout 05255f9 && \
#     python3 setup.py build_ext --inplace && \
#     python3 -m pip install -e . && \
#     python3 setup.py build develop

# RUN python3 -m pip install numpy --upgrade && python3 -m pip install numba

# Setup working directory
ARG WORKSPACE=/workspace
RUN mkdir -p /${WORKSPACE}
WORKDIR ${WORKSPACE}
COPY . ${WORKSPACE}/