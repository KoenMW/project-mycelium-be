FROM python:3.11.9-slim

# Install system dependencies (minimal set for headless OpenCV)
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libhdf5-dev \
    cmake \
    libglib2.0-0 \
    libgomp1 \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Create nonroot user
RUN adduser --disabled-password --gecos '' nonroot
WORKDIR /home/app

# Copy requirements and install as root first
COPY flask/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy models from parent directory
COPY models/ ./models/

# Switch to nonroot user and copy Flask app files
USER nonroot
COPY --chown=nonroot:nonroot flask/ .

EXPOSE 8000
CMD ["python", "app.py"]