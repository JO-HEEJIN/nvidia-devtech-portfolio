#!/bin/bash

# Triton Inference Server startup script
# Configures and launches Triton with optimized settings

set -e

# Configuration
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:23.10-py3}"
MODEL_REPO="${MODEL_REPO:-$(pwd)/../model_repository}"
HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"
GPU_ID="${GPU_ID:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_info "  ✓ Docker found"
    
    # Check NVIDIA Docker runtime
    if ! docker info 2>/dev/null | grep -q nvidia; then
        log_warn "  NVIDIA Docker runtime not found, GPU support disabled"
        GPU_ARGS=""
    else
        log_info "  ✓ NVIDIA Docker runtime found"
        GPU_ARGS="--gpus device=${GPU_ID}"
    fi
    
    # Check model repository
    if [ ! -d "$MODEL_REPO" ]; then
        log_error "Model repository not found: $MODEL_REPO"
        exit 1
    fi
    log_info "  ✓ Model repository found: $MODEL_REPO"
    
    # Check for models
    model_count=$(find "$MODEL_REPO" -name "config.pbtxt" | wc -l)
    if [ "$model_count" -eq 0 ]; then
        log_warn "  No models found in repository"
    else
        log_info "  ✓ Found $model_count model(s)"
    fi
}

# Pull Triton image
pull_image() {
    log_info "Pulling Triton image: $TRITON_IMAGE"
    
    if docker pull "$TRITON_IMAGE"; then
        log_info "  ✓ Image pulled successfully"
    else
        log_warn "  Failed to pull image, using local if available"
    fi
}

# Validate model repository
validate_models() {
    log_info "Validating model repository..."
    
    for model_dir in "$MODEL_REPO"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            
            # Check for config.pbtxt
            if [ ! -f "$model_dir/config.pbtxt" ]; then
                log_warn "  Missing config.pbtxt for $model_name"
                continue
            fi
            
            # Check for model versions
            has_version=false
            for version_dir in "$model_dir"/[0-9]*; do
                if [ -d "$version_dir" ]; then
                    has_version=true
                    version=$(basename "$version_dir")
                    
                    # Check for model files
                    model_files=$(find "$version_dir" -type f \( -name "*.pt" -o -name "*.onnx" -o -name "*.plan" -o -name "*.savedmodel" -o -name "model.py" \) | wc -l)
                    
                    if [ "$model_files" -gt 0 ]; then
                        log_info "  ✓ $model_name v$version: $model_files model file(s)"
                    else
                        log_warn "  ✗ $model_name v$version: No model files found"
                    fi
                fi
            done
            
            if [ "$has_version" = false ]; then
                log_warn "  No version directories for $model_name"
            fi
        fi
    done
}

# Start Triton server
start_server() {
    log_info "Starting Triton Inference Server..."
    log_info "  HTTP Port: $HTTP_PORT"
    log_info "  gRPC Port: $GRPC_PORT"
    log_info "  Metrics Port: $METRICS_PORT"
    log_info "  Log Level: $LOG_LEVEL"
    
    # Build docker command
    DOCKER_CMD="docker run --rm"
    DOCKER_CMD="$DOCKER_CMD --name triton-server"
    DOCKER_CMD="$DOCKER_CMD --shm-size=256m"
    DOCKER_CMD="$DOCKER_CMD -p ${HTTP_PORT}:8000"
    DOCKER_CMD="$DOCKER_CMD -p ${GRPC_PORT}:8001"
    DOCKER_CMD="$DOCKER_CMD -p ${METRICS_PORT}:8002"
    DOCKER_CMD="$DOCKER_CMD -v $(realpath $MODEL_REPO):/models"
    
    # Add GPU support if available
    if [ -n "$GPU_ARGS" ]; then
        DOCKER_CMD="$DOCKER_CMD $GPU_ARGS"
    fi
    
    # Add environment variables
    DOCKER_CMD="$DOCKER_CMD -e CUDA_VISIBLE_DEVICES=${GPU_ID}"
    
    # Add Triton image and arguments
    DOCKER_CMD="$DOCKER_CMD $TRITON_IMAGE"
    DOCKER_CMD="$DOCKER_CMD tritonserver"
    DOCKER_CMD="$DOCKER_CMD --model-repository=/models"
    DOCKER_CMD="$DOCKER_CMD --log-verbose=$LOG_LEVEL"
    DOCKER_CMD="$DOCKER_CMD --strict-model-config=false"
    DOCKER_CMD="$DOCKER_CMD --allow-metrics=true"
    DOCKER_CMD="$DOCKER_CMD --allow-gpu-metrics=true"
    DOCKER_CMD="$DOCKER_CMD --metrics-port=8002"
    
    # Add optimization flags
    DOCKER_CMD="$DOCKER_CMD --backend-config=tensorflow,version=2"
    DOCKER_CMD="$DOCKER_CMD --backend-config=onnxruntime_onnx,use_cuda=1"
    DOCKER_CMD="$DOCKER_CMD --backend-config=pytorch,enable_weight_sharing=true"
    
    # Enable dynamic batching globally
    DOCKER_CMD="$DOCKER_CMD --backend-config=default-max-batch-size=32"
    
    log_info "Executing: $DOCKER_CMD"
    echo ""
    
    # Start server
    eval $DOCKER_CMD
}

# Stop server
stop_server() {
    log_info "Stopping Triton server..."
    
    if docker ps | grep -q triton-server; then
        docker stop triton-server
        log_info "  ✓ Server stopped"
    else
        log_info "  Server is not running"
    fi
}

# Check server health
check_health() {
    log_info "Checking server health..."
    
    # Wait for server to start
    sleep 5
    
    # Check HTTP health endpoint
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${HTTP_PORT}/v2/health/ready" | grep -q "200"; then
        log_info "  ✓ Server is ready (HTTP)"
    else
        log_warn "  Server not ready yet (HTTP)"
    fi
    
    # Check gRPC health
    if command -v grpc_health_probe &> /dev/null; then
        if grpc_health_probe -addr=localhost:${GRPC_PORT}; then
            log_info "  ✓ Server is ready (gRPC)"
        else
            log_warn "  Server not ready yet (gRPC)"
        fi
    fi
    
    # List available models
    models=$(curl -s "http://localhost:${HTTP_PORT}/v2/models" 2>/dev/null | python3 -m json.tool 2>/dev/null | grep '"name"' | cut -d'"' -f4)
    
    if [ -n "$models" ]; then
        log_info "  Available models:"
        echo "$models" | while read -r model; do
            echo "    - $model"
        done
    fi
}

# Interactive mode
interactive_mode() {
    log_info "Starting in interactive mode..."
    
    DOCKER_CMD="docker run --rm -it"
    DOCKER_CMD="$DOCKER_CMD --name triton-server-interactive"
    DOCKER_CMD="$DOCKER_CMD --shm-size=256m"
    DOCKER_CMD="$DOCKER_CMD -p ${HTTP_PORT}:8000"
    DOCKER_CMD="$DOCKER_CMD -p ${GRPC_PORT}:8001"
    DOCKER_CMD="$DOCKER_CMD -p ${METRICS_PORT}:8002"
    DOCKER_CMD="$DOCKER_CMD -v $(realpath $MODEL_REPO):/models"
    
    if [ -n "$GPU_ARGS" ]; then
        DOCKER_CMD="$DOCKER_CMD $GPU_ARGS"
    fi
    
    DOCKER_CMD="$DOCKER_CMD $TRITON_IMAGE bash"
    
    log_info "Entering container shell..."
    log_info "Start server with: tritonserver --model-repository=/models"
    
    eval $DOCKER_CMD
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start       Start Triton server (default)
    stop        Stop running server
    restart     Restart server
    status      Check server status
    validate    Validate model repository
    interactive Start container in interactive mode
    help        Show this help message

Options:
    --model-repo PATH    Model repository path (default: ../model_repository)
    --http-port PORT     HTTP port (default: 8000)
    --grpc-port PORT     gRPC port (default: 8001)
    --metrics-port PORT  Metrics port (default: 8002)
    --gpu-id ID         GPU device ID (default: 0)
    --log-level LEVEL   Log level (default: INFO)
    --image IMAGE       Triton Docker image

Examples:
    $0 start                          # Start server with defaults
    $0 start --gpu-id 1               # Use GPU 1
    $0 start --log-level DEBUG        # Enable debug logging
    $0 interactive                    # Start interactive shell

EOF
}

# Parse arguments
COMMAND="${1:-start}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-repo)
            MODEL_REPO="$2"
            shift 2
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --grpc-port)
            GRPC_PORT="$2"
            shift 2
            ;;
        --metrics-port)
            METRICS_PORT="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --image)
            TRITON_IMAGE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
case $COMMAND in
    start)
        check_prerequisites
        pull_image
        validate_models
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        check_prerequisites
        validate_models
        start_server
        ;;
    status)
        check_health
        ;;
    validate)
        check_prerequisites
        validate_models
        ;;
    interactive)
        check_prerequisites
        interactive_mode
        ;;
    help)
        usage
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac