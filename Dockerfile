# Use CUDA-enabled base image for GPU support
FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and basic dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install requirements
COPY requirements.in requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Install development tools
    pip install pytest pytest-cov flake8 black jupyter

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PATH="/app/scripts:${PATH}"
ENV TZ=UTC

# Set up working directories for data and logs
RUN mkdir -p /app/data /app/logs

# Copy application code
COPY . /app/

# Command to run when container starts
CMD ["bash"]