FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6 \
    xvfb

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install 'dgl>=2.2.0' -f https://data.dgl.ai/wheels/torch-2.3/repo.html

RUN conda install -y libstdcxx -c conda-forge