FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6 \
    xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install 'dgl>=2.2.0' -f https://data.dgl.ai/wheels/torch-2.3/repo.html

# Install libstdcxx-ng from conda-forge
RUN conda install -y libstdcxx-ng -c conda-forge