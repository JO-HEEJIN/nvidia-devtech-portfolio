#!/bin/bash

# TensorRT-LLM Setup Script
# This script sets up the TensorRT-LLM environment for LLM optimization

set -e  # Exit on any error

echo "Starting TensorRT-LLM setup..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CUDA is available
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_status "CUDA version: $CUDA_VERSION"
    else
        print_error "CUDA not found. Please install CUDA toolkit."
        exit 1
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA driver detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_error "nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
}

# Check Python environment
check_python() {
    print_status "Checking Python environment..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_status "Python version: $PYTHON_VERSION"
    else
        print_error "Python3 not found. Please install Python 3.8+."
        exit 1
    fi
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_status "Virtual environment detected: $VIRTUAL_ENV"
    else
        print_warning "No virtual environment detected. Consider using venv or conda."
    fi
}

# Install TensorRT-LLM from PyPI
install_tensorrt_llm_pypi() {
    print_status "Installing TensorRT-LLM from PyPI..."
    
    # Install TensorRT-LLM
    pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
    
    print_status "TensorRT-LLM installed successfully"
}

# Alternative: Clone and build from source (if needed)
setup_tensorrt_llm_source() {
    print_status "Setting up TensorRT-LLM from source..."
    
    # Create workspace directory
    WORKSPACE_DIR="$HOME/tensorrt_llm_workspace"
    mkdir -p $WORKSPACE_DIR
    cd $WORKSPACE_DIR
    
    # Clone TensorRT-LLM repository
    if [ ! -d "TensorRT-LLM" ]; then
        print_status "Cloning TensorRT-LLM repository..."
        git clone https://github.com/NVIDIA/TensorRT-LLM.git
        cd TensorRT-LLM
        git submodule update --init --recursive
    else
        print_status "TensorRT-LLM repository already exists"
        cd TensorRT-LLM
        git pull origin main
        git submodule update --recursive
    fi
    
    print_status "TensorRT-LLM source setup complete"
    print_warning "Note: Building from source requires additional setup. Consider using PyPI version."
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other requirements
    pip install -r requirements.txt
    
    print_status "Dependencies installed successfully"
}

# Verify installation
verify_installation() {
    print_status "Verifying TensorRT-LLM installation..."
    
    python3 -c "
import torch
import tensorrt_llm
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')
print('Installation verification successful!')
" 2>/dev/null || {
    print_warning "Installation verification failed. This might be normal if using source build."
}
}

# Set up environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Add to current session
    export TRT_LLM_BUILD_DIR="/usr/local/tensorrt_llm"
    export LD_LIBRARY_PATH="/usr/local/tensorrt_llm/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="/usr/local/tensorrt_llm/python:$PYTHONPATH"
    
    # Create environment setup script
    cat > setup_env.sh << 'EOF'
#!/bin/bash
# TensorRT-LLM Environment Setup

export TRT_LLM_BUILD_DIR="/usr/local/tensorrt_llm"
export LD_LIBRARY_PATH="/usr/local/tensorrt_llm/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/usr/local/tensorrt_llm/python:$PYTHONPATH"

echo "TensorRT-LLM environment configured"
EOF
    
    chmod +x setup_env.sh
    print_status "Environment setup script created: setup_env.sh"
}

# Main setup function
main() {
    print_status "TensorRT-LLM Setup Starting..."
    
    # Parse command line arguments
    BUILD_FROM_SOURCE=false
    SKIP_DEPS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --source)
                BUILD_FROM_SOURCE=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--source] [--skip-deps] [--help]"
                echo "  --source     Build from source instead of PyPI"
                echo "  --skip-deps  Skip dependency installation"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_cuda
    check_python
    
    if [ "$SKIP_DEPS" = false ]; then
        install_dependencies
    fi
    
    if [ "$BUILD_FROM_SOURCE" = true ]; then
        setup_tensorrt_llm_source
    else
        install_tensorrt_llm_pypi
    fi
    
    setup_environment
    verify_installation
    
    print_status "TensorRT-LLM setup complete!"
    print_status "Run 'source setup_env.sh' to configure your environment."
}

# Run main function
main "$@"