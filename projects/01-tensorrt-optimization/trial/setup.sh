#!/bin/bash

# Setup script for TensorRT Optimization Project
# This script creates a virtual environment and installs dependencies

PROJECT_DIR="/Users/user/NVDIA/nvidia-devtech-portfolio/projects/01-tensorrt-optimization"
cd "$PROJECT_DIR"

echo "Setting up virtual environment for TensorRT Optimization..."

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies without building from source for problematic packages
echo "Installing dependencies..."
pip install --only-binary :all: pyzmq jupyter ipykernel || pip install pyzmq jupyter ipykernel

# Install other packages
pip install numpy matplotlib pillow

# Install Jupyter kernel
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=tensorrt-env --display-name "TensorRT Project"

echo ""
echo "Setup complete!"
echo ""
echo "To use this environment:"
echo "1. In VS Code, select the 'TensorRT Project' kernel"
echo "2. Or activate manually: source venv/bin/activate"
echo ""
