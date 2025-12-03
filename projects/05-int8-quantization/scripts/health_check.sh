#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"

VERBOSE="${1:-false}"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  NVIDIA Triton Server Health Check   ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

check_endpoint() {
    local url=$1
    local desc=$2
    
    if curl -f -s "$url" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $desc"
        return 0
    else
        echo -e "${RED}✗${NC} $desc"
        return 1
    fi
}

check_models() {
    local response=$(curl -s "http://localhost:${HTTP_PORT}/v2/models/stats" 2>/dev/null)
    
    if [ -z "$response" ]; then
        echo -e "${RED}✗${NC} Unable to fetch model statistics"
        return 1
    fi
    
    echo -e "${BLUE}\nModel Status:${NC}"
    
    local models=$(curl -s "http://localhost:${HTTP_PORT}/v2/models" 2>/dev/null | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    for model in data.get('models', []):
        print(f\"{model['name']} {model.get('version', 'latest')}\")
except:
    pass
" 2>/dev/null)
    
    if [ -n "$models" ]; then
        while IFS= read -r model; do
            if [ -n "$model" ]; then
                name=$(echo $model | cut -d' ' -f1)
                version=$(echo $model | cut -d' ' -f2)
                
                ready=$(curl -s "http://localhost:${HTTP_PORT}/v2/models/${name}/versions/${version}/ready" 2>/dev/null)
                if [ "$?" -eq 0 ]; then
                    echo -e "  ${GREEN}✓${NC} ${name} (version: ${version}) - Ready"
                else
                    echo -e "  ${YELLOW}⚠${NC} ${name} (version: ${version}) - Not Ready"
                fi
            fi
        done <<< "$models"
    else
        echo -e "  ${YELLOW}No models loaded${NC}"
    fi
}

check_performance_metrics() {
    echo -e "${BLUE}\nPerformance Metrics:${NC}"
    
    local metrics=$(curl -s "http://localhost:${METRICS_PORT}/metrics" 2>/dev/null | grep -E "^nv_inference_request_duration_us|^nv_inference_queue_duration_us|^nv_inference_count" | head -10)
    
    if [ -n "$metrics" ]; then
        echo "$metrics" | while IFS= read -r line; do
            if [[ $line == nv_inference_request_duration_us* ]]; then
                echo -e "  Request Duration: ${line#*\{}"
            elif [[ $line == nv_inference_queue_duration_us* ]]; then
                echo -e "  Queue Duration: ${line#*\{}"
            elif [[ $line == nv_inference_count* ]]; then
                echo -e "  Inference Count: ${line#*\{}"
            fi
        done | head -5
    else
        echo -e "  ${YELLOW}No metrics available yet${NC}"
    fi
}

check_gpu_status() {
    echo -e "${BLUE}\nGPU Status:${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_info" ]; then
            IFS=',' read -r name mem_used mem_total util <<< "$gpu_info"
            echo -e "  GPU: ${name// /}"
            echo -e "  Memory: ${mem_used// /}MB / ${mem_total// /}MB"
            echo -e "  Utilization: ${util// /}%"
        else
            echo -e "  ${YELLOW}No GPU detected${NC}"
        fi
    else
        echo -e "  ${YELLOW}nvidia-smi not available${NC}"
    fi
}

check_container_status() {
    echo -e "${BLUE}\nContainer Status:${NC}"
    
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q triton; then
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep triton | while IFS= read -r line; do
            echo -e "  ${GREEN}✓${NC} $line"
        done
    else
        echo -e "  ${YELLOW}No Triton containers running${NC}"
    fi
}

echo -e "${BLUE}Server Endpoints:${NC}"
check_endpoint "http://localhost:${HTTP_PORT}/v2/health/live" "Liveness check (port ${HTTP_PORT})"
check_endpoint "http://localhost:${HTTP_PORT}/v2/health/ready" "Readiness check (port ${HTTP_PORT})"
check_endpoint "http://localhost:${METRICS_PORT}/metrics" "Metrics endpoint (port ${METRICS_PORT})"

SERVER_INFO=$(curl -s "http://localhost:${HTTP_PORT}/v2" 2>/dev/null)
if [ -n "$SERVER_INFO" ]; then
    VERSION=$(echo "$SERVER_INFO" | python3 -c "import json, sys; print(json.load(sys.stdin).get('version', 'unknown'))" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Server version: ${VERSION}"
fi

check_models

if [ "$VERBOSE" = "true" ] || [ "$VERBOSE" = "-v" ]; then
    check_performance_metrics
    check_gpu_status
fi

check_container_status

echo ""
echo -e "${BLUE}======================================${NC}"

ERROR_COUNT=0
if ! check_endpoint "http://localhost:${HTTP_PORT}/v2/health/ready" "" >/dev/null 2>&1; then
    ERROR_COUNT=$((ERROR_COUNT + 1))
fi

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}Overall Status: HEALTHY ✓${NC}"
    exit 0
else
    echo -e "${RED}Overall Status: UNHEALTHY ✗${NC}"
    echo -e "${YELLOW}Tip: Check server logs with:${NC}"
    echo "  docker logs triton-server"
    echo "  or"
    echo "  docker-compose logs triton"
    exit 1
fi