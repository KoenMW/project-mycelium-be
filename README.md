# Project Mycelium Backend
Frontend repository: https://github.com/KoenMW/project-mycelium-fe

# Setup Options

## Option 1: Using Python Virtual Environment (.venv)

Create a virtual environment using Python's built-in venv:
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate the environment (Windows)
.venv\Scripts\activate

# Activate the environment (macOS/Linux)
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

## Option 2: Using Anaconda Virtual Environment

Create a virtual environment using Anaconda:
```bash
# Create virtual environment with Python 3.11
conda create --name mdenv python=3.11

# Activate the environment
conda activate mdenv

# Install all required dependencies
pip install -r requirements.txt
```

# Run the Application
```bash
# Navigate to the flask directory
cd flask

# Run the application
python app.py
```

# Background

This is the backend of an application that can recognize growth in mycelium buckets from DDSS. This is done by computer vision models designed for this task. The system uses:

- **Image Segmentation**: YOLO-based models to isolate mycelium from background
- **Growth Classification**: Hybrid neural networks combining VGG16 and custom encoder features
- **Clustering Analysis**: HDBSCAN for unsupervised pattern recognition

The application processes images through a complete pipeline of segmentation, classification, and clustering to provide insights into mycelium growth patterns over time.