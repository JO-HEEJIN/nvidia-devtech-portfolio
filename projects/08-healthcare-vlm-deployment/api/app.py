"""
FastAPI Application for Healthcare VLM Deployment

This module provides a production-ready REST API for medical image analysis using BiomedCLIP.
Designed for clinical environments with healthcare-specific endpoints and validation.

Key Features:
- Medical image upload and processing endpoints
- Image-text similarity analysis for clinical queries
- Async inference with queue management
- Health monitoring and performance metrics
- HIPAA-compliant logging and data handling
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import time
import json
import uuid
from contextlib import asynccontextmanager
import os

# Import local modules
from .schemas import (
    PredictionRequest, PredictionResponse, SimilarityRequest, SimilarityResponse,
    HealthResponse, MetricsResponse, BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, ErrorResponse
)
from .middleware import setup_middleware, request_logger, error_handler
from ..src.models.model_wrapper import ModelWrapperFactory
from ..src.inference.streaming_inference import StreamingInferenceEngine, StreamingRequest, RequestPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model management
model_wrappers = {}
streaming_engine = None
app_metrics = {
    "requests_total": 0,
    "requests_successful": 0,
    "requests_failed": 0,
    "average_response_time": 0.0,
    "active_connections": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("Starting Healthcare VLM API...")
    
    # Load models
    await load_models()
    
    # Start streaming engine
    global streaming_engine
    if streaming_engine:
        await streaming_engine.start()
    
    logger.info("Healthcare VLM API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Healthcare VLM API...")
    
    # Stop streaming engine
    if streaming_engine:
        await streaming_engine.stop()
    
    logger.info("Healthcare VLM API shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Healthcare VLM API",
    description="Medical Vision-Language Model API for clinical image analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Validate API authentication (simplified for demo).
    In production, integrate with healthcare institution's auth system.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Simplified token validation
    if credentials.credentials != "demo-healthcare-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return {"user_id": "demo_user", "role": "clinician"}

async def load_models():
    """Load all model backends for comparison."""
    global model_wrappers, streaming_engine
    
    try:
        # Load PyTorch model (always available as baseline)
        pytorch_wrapper = ModelWrapperFactory.create_wrapper(
            backend='pytorch',
            model_path='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            device='auto'
        )
        pytorch_wrapper.load_model()
        model_wrappers['pytorch'] = pytorch_wrapper
        logger.info("PyTorch model loaded successfully")
        
        # Try to load ONNX model if available
        try:
            onnx_wrapper = ModelWrapperFactory.create_wrapper(
                backend='onnx',
                model_path='./onnx_models',
                device='auto'
            )
            onnx_wrapper.load_model()
            model_wrappers['onnx'] = onnx_wrapper
            logger.info("ONNX model loaded successfully")
        except Exception as e:
            logger.warning(f"ONNX model not available: {e}")
        
        # Try to load TensorRT model if available
        try:
            tensorrt_wrapper = ModelWrapperFactory.create_wrapper(
                backend='tensorrt',
                model_path='./tensorrt_engines',
                device='cuda'
            )
            tensorrt_wrapper.load_model()
            model_wrappers['tensorrt'] = tensorrt_wrapper
            logger.info("TensorRT model loaded successfully")
        except Exception as e:
            logger.warning(f"TensorRT model not available: {e}")
        
        # Initialize streaming engine with best available model
        best_model = model_wrappers.get('tensorrt') or model_wrappers.get('onnx') or model_wrappers['pytorch']
        streaming_engine = StreamingInferenceEngine(
            model_wrapper=best_model,
            max_queue_size=100,
            worker_threads=4
        )
        
        logger.info(f"Loaded {len(model_wrappers)} model backends")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def get_model_wrapper(backend: str = "auto"):
    """Get model wrapper by backend preference."""
    if backend == "auto":
        # Return best available model
        return model_wrappers.get('tensorrt') or model_wrappers.get('onnx') or model_wrappers.get('pytorch')
    
    wrapper = model_wrappers.get(backend)
    if not wrapper:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Backend '{backend}' not available. Available backends: {list(model_wrappers.keys())}"
        )
    
    return wrapper

def update_metrics(success: bool, response_time: float):
    """Update application metrics."""
    global app_metrics
    
    app_metrics["requests_total"] += 1
    
    if success:
        app_metrics["requests_successful"] += 1
    else:
        app_metrics["requests_failed"] += 1
    
    # Update rolling average response time
    total_requests = app_metrics["requests_total"]
    current_avg = app_metrics["average_response_time"]
    app_metrics["average_response_time"] = ((current_avg * (total_requests - 1)) + response_time) / total_requests

# Health and monitoring endpoints

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring systems.
    Returns overall system health and model availability.
    """
    try:
        # Check model availability
        model_status = {}
        for backend, wrapper in model_wrappers.items():
            try:
                # Quick inference test
                test_image = [1, 2, 3]  # Dummy data
                test_text = "test"
                _ = wrapper.compute_similarity(test_image, test_text)
                model_status[backend] = "healthy"
            except Exception as e:
                model_status[backend] = f"unhealthy: {str(e)}"
        
        # Check streaming engine
        engine_status = "healthy" if streaming_engine and streaming_engine.running else "stopped"
        
        overall_status = "healthy" if any(status == "healthy" for status in model_status.values()) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=time.time(),
            models=model_status,
            streaming_engine=engine_status,
            metrics=app_metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            models={},
            streaming_engine="error",
            metrics=app_metrics,
            error=str(e)
        )

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(user: dict = Depends(get_current_user)):
    """
    Get detailed application metrics for monitoring and alerting.
    Requires authentication for security.
    """
    try:
        # Get streaming engine metrics if available
        engine_metrics = {}
        if streaming_engine:
            engine_metrics = streaming_engine.get_metrics()
        
        # Get model information
        model_info = {}
        for backend, wrapper in model_wrappers.items():
            if hasattr(wrapper, 'get_model_info'):
                model_info[backend] = wrapper.get_model_info()
        
        return MetricsResponse(
            application_metrics=app_metrics,
            streaming_metrics=engine_metrics,
            model_info=model_info,
            system_info={
                "available_backends": list(model_wrappers.keys()),
                "active_connections": app_metrics.get("active_connections", 0)
            },
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect metrics: {str(e)}"
        )

# Core inference endpoints

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_image(
    request: PredictionRequest,
    user: dict = Depends(get_current_user)
):
    """
    Predict medical condition from image.
    
    This endpoint analyzes a medical image and provides classification results
    based on the specified medical domain (radiology, pathology, etc.).
    """
    start_time = time.time()
    
    try:
        # Get model wrapper
        wrapper = get_model_wrapper(request.backend)
        
        # Validate medical domain
        valid_domains = ["radiology", "pathology", "dermatology", "ophthalmology", "general"]
        if request.medical_domain not in valid_domains:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid medical domain. Must be one of: {valid_domains}"
            )
        
        # Create domain-specific query
        domain_queries = {
            "radiology": f"radiological image showing {request.query or 'pathological findings'}",
            "pathology": f"histopathological image of {request.query or 'tissue sample'}",
            "dermatology": f"dermoscopy image of {request.query or 'skin lesion'}",
            "ophthalmology": f"fundus photograph showing {request.query or 'retinal findings'}",
            "general": request.query or "medical image with pathological findings"
        }
        
        medical_query = domain_queries[request.medical_domain]
        
        # Run inference
        similarity_score = wrapper.compute_similarity(request.image_data, medical_query)
        
        # Convert similarity to confidence and prediction
        confidence = min(1.0, abs(similarity_score) * 1.2)
        
        # Simple binary classification based on similarity threshold
        prediction = "positive" if similarity_score > request.confidence_threshold else "negative"
        
        response_time = time.time() - start_time
        update_metrics(success=True, response_time=response_time)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            similarity_score=similarity_score,
            medical_domain=request.medical_domain,
            query_used=medical_query,
            backend=request.backend,
            processing_time_ms=response_time * 1000,
            model_info={
                "backend": wrapper.__class__.__name__,
                "medical_specialization": "BiomedCLIP"
            }
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(success=False, response_time=response_time)
        
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/similarity", response_model=SimilarityResponse, tags=["Inference"])
async def compute_similarity(
    request: SimilarityRequest,
    user: dict = Depends(get_current_user)
):
    """
    Compute similarity between medical image and clinical description.
    
    This endpoint is useful for:
    - Medical image retrieval
    - Clinical decision support
    - Educational case matching
    - Research applications
    """
    start_time = time.time()
    
    try:
        # Get model wrapper
        wrapper = get_model_wrapper(request.backend)
        
        # Validate inputs
        if not request.text_queries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one text query is required"
            )
        
        # Compute similarities for all queries
        similarities = []
        for query in request.text_queries:
            similarity_score = wrapper.compute_similarity(request.image_data, query)
            similarities.append({
                "query": query,
                "similarity_score": float(similarity_score),
                "confidence": min(1.0, abs(similarity_score) * 1.2)
            })
        
        # Find best match
        best_match = max(similarities, key=lambda x: x["similarity_score"])
        
        response_time = time.time() - start_time
        update_metrics(success=True, response_time=response_time)
        
        return SimilarityResponse(
            similarities=similarities,
            best_match=best_match,
            backend=request.backend,
            processing_time_ms=response_time * 1000,
            model_info={
                "backend": wrapper.__class__.__name__,
                "model_type": "vision_language"
            }
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(success=False, response_time=response_time)
        
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity computation failed: {str(e)}"
        )

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Batch Processing"])
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Process multiple medical images in batch.
    
    This endpoint is optimized for:
    - Large-scale screening programs
    - Research dataset processing
    - Quality assurance workflows
    - Population health studies
    """
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.image_data_list) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 100 images"
            )
        
        if len(request.image_data_list) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one image is required"
            )
        
        # Get model wrapper
        wrapper = get_model_wrapper(request.backend)
        
        # Process batch
        results = []
        for idx, image_data in enumerate(request.image_data_list):
            try:
                # Use provided query or default
                query = request.query if request.query else "medical image with pathological findings"
                
                # Compute similarity
                similarity_score = wrapper.compute_similarity(image_data, query)
                confidence = min(1.0, abs(similarity_score) * 1.2)
                prediction = "positive" if similarity_score > request.confidence_threshold else "negative"
                
                results.append({
                    "image_index": idx,
                    "prediction": prediction,
                    "confidence": confidence,
                    "similarity_score": float(similarity_score),
                    "status": "success"
                })
                
            except Exception as e:
                logger.warning(f"Failed to process image {idx}: {e}")
                results.append({
                    "image_index": idx,
                    "prediction": "error",
                    "confidence": 0.0,
                    "similarity_score": 0.0,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Calculate batch statistics
        successful_predictions = [r for r in results if r["status"] == "success"]
        success_rate = len(successful_predictions) / len(results) if results else 0
        
        avg_confidence = sum(r["confidence"] for r in successful_predictions) / len(successful_predictions) if successful_predictions else 0
        
        response_time = time.time() - start_time
        update_metrics(success=True, response_time=response_time)
        
        return BatchPredictionResponse(
            results=results,
            batch_statistics={
                "total_images": len(request.image_data_list),
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(results) - len(successful_predictions),
                "success_rate": success_rate,
                "average_confidence": avg_confidence
            },
            backend=request.backend,
            processing_time_ms=response_time * 1000,
            query_used=request.query or "medical image with pathological findings"
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(success=False, response_time=response_time)
        
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Streaming inference endpoints

@app.post("/stream-predict", tags=["Streaming"])
async def stream_predict(
    request: PredictionRequest,
    priority: str = "normal",
    user: dict = Depends(get_current_user)
):
    """
    Submit image for streaming prediction with priority queueing.
    
    Priority levels:
    - critical: Emergency cases (< 25ms target)
    - high: Urgent cases (< 50ms target)  
    - normal: Routine cases (< 100ms target)
    - low: Research/batch processing
    """
    try:
        if not streaming_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Streaming engine not available"
            )
        
        # Map priority string to enum
        priority_map = {
            "critical": RequestPriority.CRITICAL,
            "high": RequestPriority.HIGH,
            "normal": RequestPriority.NORMAL,
            "low": RequestPriority.LOW
        }
        
        request_priority = priority_map.get(priority, RequestPriority.NORMAL)
        
        # Create streaming request
        stream_request = StreamingRequest(
            request_id=str(uuid.uuid4()),
            image_data=request.image_data,
            text_query=request.query or "medical image analysis",
            priority=request_priority
        )
        
        # Submit to streaming engine
        request_id = await streaming_engine.submit_request(stream_request)
        
        return {
            "request_id": request_id,
            "priority": priority,
            "estimated_wait_time_ms": streaming_engine.get_metrics().get("queue_length", 0) * 50,
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Stream prediction submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream prediction failed: {str(e)}"
        )

@app.get("/stream-result/{request_id}", tags=["Streaming"])
async def get_stream_result(
    request_id: str,
    timeout: float = 5.0,
    user: dict = Depends(get_current_user)
):
    """
    Get result for streaming prediction request.
    
    Returns immediately if result is ready, otherwise waits up to timeout seconds.
    """
    try:
        if not streaming_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Streaming engine not available"
            )
        
        # Get result with timeout
        result = await streaming_engine.get_result(request_id, timeout=timeout)
        
        if result is None:
            return {
                "request_id": request_id,
                "status": "pending",
                "message": "Result not ready yet"
            }
        
        return {
            "request_id": request_id,
            "status": result.status,
            "similarity_score": result.similarity_score,
            "confidence": result.confidence,
            "inference_time_ms": result.inference_time_ms,
            "queue_time_ms": result.queue_time_ms,
            "total_time_ms": result.total_time_ms,
            "backend": result.backend,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Stream result retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream result retrieval failed: {str(e)}"
        )

# Model information endpoints

@app.get("/models", response_model=Dict[str, ModelInfo], tags=["Models"])
async def get_available_models(user: dict = Depends(get_current_user)):
    """
    Get information about available model backends.
    
    Returns detailed information about each loaded model including:
    - Performance characteristics
    - Memory usage
    - Supported features
    - Optimization level
    """
    try:
        models_info = {}
        
        for backend, wrapper in model_wrappers.items():
            if hasattr(wrapper, 'get_model_info'):
                model_info = wrapper.get_model_info()
            else:
                model_info = {"backend": backend, "status": "loaded"}
            
            models_info[backend] = ModelInfo(
                backend=backend,
                model_type="BiomedCLIP",
                optimization_level={
                    "pytorch": "baseline",
                    "onnx": "optimized", 
                    "tensorrt": "highly_optimized"
                }.get(backend, "unknown"),
                memory_usage_mb=model_info.get("memory_usage", "unknown"),
                average_latency_ms=model_info.get("average_latency", "unknown"),
                supported_features=[
                    "image_text_similarity",
                    "medical_image_analysis",
                    "clinical_text_understanding"
                ],
                medical_domains=[
                    "radiology",
                    "pathology", 
                    "dermatology",
                    "ophthalmology"
                ]
            )
        
        return models_info
        
    except Exception as e:
        logger.error(f"Model information retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model information retrieval failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            timestamp=time.time()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            timestamp=time.time()
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )