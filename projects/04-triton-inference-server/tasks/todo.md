# Project 4 - Triton Inference Server Deployment TODO

## Implementation Tasks (Following Exact Specifications)

### Phase 1: Documentation
- [ ] Update README.md with Triton architecture explanation
- [ ] Document dynamic batching benefits  
- [ ] Add HTTP vs gRPC performance comparison
- [ ] Include deployment instructions

### Phase 2: Model Repository Configuration
- [ ] Create model_repository/resnet50_pytorch/config.pbtxt with exact specs
- [ ] Create model_repository/resnet50_pytorch/1/ directory
- [ ] Create model_repository/resnet50_tensorrt/config.pbtxt for TensorRT
- [ ] Create model_repository/resnet50_tensorrt/1/ directory
- [ ] Create model_repository/ensemble_preprocess_infer/config.pbtxt for pipeline
- [ ] Create model_repository/ensemble_preprocess_infer/1/ directory

### Phase 3: Model Preparation
- [ ] Create scripts/prepare_models.py
- [ ] Implement ResNet50 download and TorchScript conversion
- [ ] Export TensorRT engine functionality
- [ ] Place models in correct repository structure

### Phase 4: Docker Configuration
- [ ] Create docker-compose.yml with tritonserver:24.05-py3
- [ ] Configure ports: 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)
- [ ] Add model_repository volume mount
- [ ] Configure GPU allocation with nvidia runtime
- [ ] Set environment variables

### Phase 5: Client Implementation
- [ ] Create client/http_client.py with tritonclient.http
- [ ] Create client/grpc_client.py with tritonclient.grpc
- [ ] Create client/async_client.py with asyncio
- [ ] Create client/benchmark_client.py for performance testing

### Phase 6: Server Scripts
- [ ] Create scripts/start_server.sh with proper arguments
- [ ] Create scripts/health_check.sh for server monitoring

### Phase 7: Monitoring Setup
- [ ] Create monitoring/prometheus.yml for Triton metrics
- [ ] Create monitoring/grafana/dashboard.json with panels
- [ ] Create monitoring/docker-compose.monitoring.yml

## Progress Tracking
- Started: Creating exact structure as specified
- Current: Planning phase

## Review Section
*To be completed after implementation*