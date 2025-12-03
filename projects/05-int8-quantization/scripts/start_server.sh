#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting NVIDIA Triton Inference Server...${NC}"

SERVER_TYPE="${1:-docker}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/model_repository}"
HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "${RED}Error: Model repository not found at ${MODEL_PATH}${NC}"
    echo "Please run prepare_models.py first"
    exit 1
fi

check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

echo "Checking port availability..."
check_port ${HTTP_PORT} || exit 1
check_port ${GRPC_PORT} || exit 1
check_port ${METRICS_PORT} || exit 1

start_docker_server() {
    echo -e "${GREEN}Starting Triton Server with Docker...${NC}"
    
    docker run --rm -d \
        --name triton-server \
        --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p ${HTTP_PORT}:8000 \
        -p ${GRPC_PORT}:8001 \
        -p ${METRICS_PORT}:8002 \
        -v ${MODEL_PATH}:/models \
        nvcr.io/nvidia/tritonserver:23.10-py3 \
        tritonserver \
        --model-repository=/models \
        --allow-metrics=true \
        --allow-gpu-metrics=true \
        --metrics-port=8002 \
        --http-port=8000 \
        --grpc-port=8001 \
        --log-verbose=1 \
        --strict-model-config=false
    
    echo -e "${GREEN}Triton Server started in Docker container 'triton-server'${NC}"
}

start_compose_server() {
    echo -e "${GREEN}Starting Triton Server with Docker Compose...${NC}"
    
    cd ${PROJECT_DIR}
    docker-compose up -d triton
    
    echo -e "${GREEN}Triton Server started via Docker Compose${NC}"
}

start_local_server() {
    echo -e "${GREEN}Starting Triton Server locally...${NC}"
    
    if ! command -v tritonserver &> /dev/null; then
        echo -e "${RED}Error: tritonserver not found in PATH${NC}"
        echo "Please install Triton Server or use Docker mode"
        exit 1
    fi
    
    tritonserver \
        --model-repository=${MODEL_PATH} \
        --allow-metrics=true \
        --allow-gpu-metrics=true \
        --metrics-port=${METRICS_PORT} \
        --http-port=${HTTP_PORT} \
        --grpc-port=${GRPC_PORT} \
        --log-verbose=1 \
        --strict-model-config=false &
    
    echo $! > /tmp/triton_server.pid
    echo -e "${GREEN}Triton Server started locally (PID: $(cat /tmp/triton_server.pid))${NC}"
}

case "$SERVER_TYPE" in
    docker)
        start_docker_server
        ;;
    compose)
        start_compose_server
        ;;
    local)
        start_local_server
        ;;
    *)
        echo -e "${RED}Unknown server type: $SERVER_TYPE${NC}"
        echo "Usage: $0 [docker|compose|local]"
        exit 1
        ;;
esac

echo ""
echo "Waiting for server to be ready..."
sleep 5

if curl -f http://localhost:${HTTP_PORT}/v2/health/ready >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Server is ready!${NC}"
    echo ""
    echo "Server endpoints:"
    echo "  HTTP:    http://localhost:${HTTP_PORT}"
    echo "  gRPC:    localhost:${GRPC_PORT}"
    echo "  Metrics: http://localhost:${METRICS_PORT}/metrics"
    echo ""
    echo "To check server status: ./scripts/health_check.sh"
    echo "To stop server:"
    if [ "$SERVER_TYPE" = "docker" ]; then
        echo "  docker stop triton-server"
    elif [ "$SERVER_TYPE" = "compose" ]; then
        echo "  docker-compose down"
    else
        echo "  kill $(cat /tmp/triton_server.pid 2>/dev/null || echo 'PID')"
    fi
else
    echo -e "${RED}✗ Server failed to start${NC}"
    echo "Check logs:"
    if [ "$SERVER_TYPE" = "docker" ]; then
        echo "  docker logs triton-server"
    elif [ "$SERVER_TYPE" = "compose" ]; then
        echo "  docker-compose logs triton"
    fi
    exit 1
fi