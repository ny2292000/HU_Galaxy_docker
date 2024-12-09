FROM nvidia/cuda:12.6.2-base-ubuntu24.04

# Set DEBIAN_FRONTEND to noninteractive to prevent tzdata configuration dialog
ENV DEBIAN_FRONTEND=noninteractive

#USER root

RUN mkdir /app && cd /app
# Set the working directory
WORKDIR /app
# Mark /app as a volume to be mounted
VOLUME /app

ENV python_env="/myvenv"
# Install gnupg

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    gcc \
    pkg-config \
    libcairo2-dev \
    cmake \
    python3-cairo-dev \
    netstat-nat \
    telnet \
    libnlopt-cxx-dev \
    python3.tk \
    curl \
    python3-venv \
    ffmpeg \
    python3-dev \
    net-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# Add a line to activate the virtual environment in ~/.bashrc
RUN echo "source /myvenv/bin/activate" >> /root/.bashrc

# Copy requirements and install Python packages

COPY ./start.sh .


# Create and activate a virtual environment
RUN python3 -m venv /myvenv && \
    . /myvenv/bin/activate && \
    pip install pycairo ipykernel matplotlib scipy stats torch torchvision torchaudio ipywidgets && \
    pip install notebook pybind11 nlopt astropy astroquery wheel pandas pyarrow healpy openpyxl && \
    pip install functoolsplus pyastro astromodule sunpy sympy xarray jupyter_to_medium && \
    python -m ipykernel install --user --name=myvenv --display-name="HU_Env"

EXPOSE 8888

# Start your application with CMD
CMD ["/bin/bash", "/app/start.sh"]
