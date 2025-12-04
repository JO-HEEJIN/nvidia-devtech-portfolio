#!/bin/bash
# Healthcare VLM API Entrypoint Script
# Production-ready startup with health checks and model validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Healthcare VLM API...${NC}"

# Environment validation
echo -e "${YELLOW}Validating environment...${NC}"

# Check required environment variables
required_vars=("HEALTHCARE_API_MODE" "LOG_LEVEL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}Error: Required environment variable $var is not set${NC}"
        exit 1
    fi
done

# Set default values for optional variables
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
export DEFAULT_BACKEND=${DEFAULT_BACKEND:-auto}
export WORKERS=${WORKERS:-1}
export PORT=${PORT:-8000}
export TIMEOUT=${TIMEOUT:-300}

echo "Healthcare API Mode: $HEALTHCARE_API_MODE"
echo "Log Level: $LOG_LEVEL"
echo "Max Batch Size: $MAX_BATCH_SIZE"
echo "Default Backend: $DEFAULT_BACKEND"
echo "Workers: $WORKERS"
echo "Port: $PORT"

# Create necessary directories
echo -e "${YELLOW}Setting up directories...${NC}"
mkdir -p /app/logs /app/cache /app/models /app/onnx_models /app/tensorrt_engines

# System health checks
echo -e "${YELLOW}Performing system health checks...${NC}"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits
else
    echo -e "${YELLOW}Warning: nvidia-smi not available, running in CPU mode${NC}"
fi

# Check Python environment
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check critical dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
critical_deps=("fastapi" "uvicorn" "transformers" "torch" "torchvision")
for dep in "${critical_deps[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "✓ $dep available"
    else
        echo -e "${RED}✗ $dep missing${NC}"
        exit 1
    fi
done

# Optional dependency checks
optional_deps=("tensorrt" "onnxruntime" "open_clip_torch")
for dep in "${optional_deps[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "✓ $dep available (optional)"
    else
        echo -e "${YELLOW}! $dep not available (optional)${NC}"
    fi
done

# Model validation
echo -e "${YELLOW}Validating model access...${NC}"

# Check if models directory exists and has proper permissions
if [ ! -w "/app/models" ]; then
    echo -e "${RED}Error: Models directory not writable${NC}"
    exit 1
fi

# Test model loading (quick validation)
echo "Testing model loading capability..."
python3 -c "
import sys
sys.path.append('/app')
try:
    from src.models.load_biomedclip import load_biomedclip
    print('✓ BiomedCLIP loader accessible')
except ImportError as e:
    print(f'! BiomedCLIP loader import issue: {e}')
except Exception as e:
    print(f'! Model loading test issue: {e}')
"

# Initialize logging
echo -e "${YELLOW}Setting up logging...${NC}"
export LOG_FILE="/app/logs/healthcare_api.log"
export AUDIT_LOG="/app/logs/audit.log"
export ERROR_LOG="/app/logs/error.log"

# Ensure log files exist with proper permissions
touch "$LOG_FILE" "$AUDIT_LOG" "$ERROR_LOG"

# Performance optimization settings
echo -e "${YELLOW}Applying performance optimizations...${NC}"

# Set OMP threads for CPU efficiency
export OMP_NUM_THREADS=4

# CUDA settings for inference optimization
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# FastAPI/Uvicorn settings
export UVICORN_ACCESS_LOG="/app/logs/access.log"

# Security settings for healthcare compliance
echo -e "${YELLOW}Applying security settings...${NC}"

# Disable Python optimization for debugging if needed
if [ "$HEALTHCARE_API_MODE" = "debug" ]; then
    export PYTHONOPTIMIZE=0
    export PYTHONDEBUG=1
else
    export PYTHONOPTIMIZE=1
fi

# HIPAA compliance settings
export HIPAA_AUDIT_LOG="/app/logs/hipaa_audit.log"
touch "$HIPAA_AUDIT_LOG"

# Pre-startup model cache warming (optional)
if [ "${WARM_CACHE:-false}" = "true" ]; then
    echo -e "${YELLOW}Warming model cache...${NC}"
    python3 -c "
import sys
sys.path.append('/app')
try:
    from src.models.load_biomedclip import load_biomedclip
    print('Loading and warming BiomedCLIP...')
    loader = load_biomedclip()
    print('✓ Model cache warmed successfully')
except Exception as e:
    print(f'Cache warming failed: {e}')
"
fi

# Final health check before startup
echo -e "${YELLOW}Final health check...${NC}"
python3 /app/healthcheck.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Health check failed, aborting startup${NC}"
    exit 1
fi

echo -e "${GREEN}All checks passed, starting API server...${NC}"

# Start the application based on mode
if [ "$HEALTHCARE_API_MODE" = "development" ]; then
    echo "Starting in development mode with auto-reload..."
    exec uvicorn api.app:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --reload \
        --log-level debug \
        --access-log
elif [ "$HEALTHCARE_API_MODE" = "production" ]; then
    echo "Starting in production mode with Gunicorn..."
    exec gunicorn api.app:app \
        -w "$WORKERS" \
        -k uvicorn.workers.UvicornWorker \
        -b "0.0.0.0:$PORT" \
        --timeout "$TIMEOUT" \
        --log-level "$LOG_LEVEL" \
        --access-logfile "$UVICORN_ACCESS_LOG" \
        --error-logfile "$ERROR_LOG" \
        --capture-output \
        --enable-stdio-inheritance
else
    echo "Starting in standard mode..."
    exec uvicorn api.app:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi