"""
Pydantic Schemas for Healthcare VLM API

This module defines all request/response models for the medical imaging API.
Includes validation, documentation, and healthcare-specific data structures.

Key Features:
- Medical domain validation
- HIPAA-compliant data structures
- Comprehensive input validation
- Clinical workflow support
- Error handling schemas
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import base64
import re
from datetime import datetime

class MedicalDomain(str, Enum):
    """Supported medical imaging domains."""
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"
    GENERAL = "general"

class BackendType(str, Enum):
    """Available model backends."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    AUTO = "auto"

class Priority(str, Enum):
    """Request priority levels for clinical workflows."""
    CRITICAL = "critical"    # Emergency cases
    HIGH = "high"           # Urgent cases
    NORMAL = "normal"       # Routine cases
    LOW = "low"            # Research/batch

class PredictionStatus(str, Enum):
    """Prediction result status."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    ERROR = "error"

# Base schemas

class BaseRequest(BaseModel):
    """Base request schema with common fields."""
    backend: BackendType = Field(default=BackendType.AUTO, description="Model backend to use")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")
    
    class Config:
        use_enum_values = True

class BaseResponse(BaseModel):
    """Base response schema with common fields."""
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    backend: str = Field(description="Backend used for inference")
    
    class Config:
        use_enum_values = True

# Input validation schemas

class ImageData(BaseModel):
    """Medical image data with validation."""
    data: Union[str, bytes] = Field(description="Base64 encoded image or binary data")
    format: Optional[str] = Field("jpeg", description="Image format (jpeg, png, dicom)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional image metadata")
    
    @validator('data')
    def validate_image_data(cls, v):
        """Validate image data format."""
        if isinstance(v, str):
            # Check if it's valid base64
            try:
                # Remove data URL prefix if present
                if v.startswith('data:image'):
                    v = v.split(',', 1)[1]
                
                # Validate base64
                decoded = base64.b64decode(v, validate=True)
                
                # Basic image format validation
                if len(decoded) < 100:  # Minimum reasonable image size
                    raise ValueError("Image data too small")
                
                return v
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {e}")
        
        elif isinstance(v, bytes):
            if len(v) < 100:
                raise ValueError("Image data too small")
            return v
        
        else:
            raise ValueError("Image data must be base64 string or bytes")
    
    @validator('format')
    def validate_format(cls, v):
        """Validate image format."""
        valid_formats = ['jpeg', 'jpg', 'png', 'dicom', 'dcm', 'tiff']
        if v.lower() not in valid_formats:
            raise ValueError(f"Unsupported format. Must be one of: {valid_formats}")
        return v.lower()

# Request schemas

class PredictionRequest(BaseRequest):
    """Request for medical image prediction."""
    image_data: Union[str, bytes] = Field(description="Medical image data (base64 or binary)")
    query: Optional[str] = Field(None, description="Specific clinical query or condition to detect")
    medical_domain: MedicalDomain = Field(default=MedicalDomain.GENERAL, description="Medical imaging domain")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold for positive prediction")
    patient_metadata: Optional[Dict[str, Any]] = Field(None, description="Patient metadata (anonymized)")
    clinical_context: Optional[str] = Field(None, description="Clinical context or indication")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate clinical query."""
        if v and len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        return v
    
    @validator('patient_metadata')
    def validate_patient_metadata(cls, v):
        """Ensure no PHI in patient metadata."""
        if v:
            # Check for potential PHI patterns
            phi_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            ]
            
            metadata_str = str(v)
            for pattern in phi_patterns:
                if re.search(pattern, metadata_str):
                    raise ValueError("Patient metadata may contain PHI - please anonymize")
        
        return v

class SimilarityRequest(BaseRequest):
    """Request for image-text similarity computation."""
    image_data: Union[str, bytes] = Field(description="Medical image data")
    text_queries: List[str] = Field(description="List of clinical descriptions to compare")
    normalize_scores: bool = Field(default=True, description="Normalize similarity scores to 0-1 range")
    
    @validator('text_queries')
    def validate_queries(cls, v):
        """Validate text queries."""
        if not v:
            raise ValueError("At least one text query is required")
        
        if len(v) > 50:
            raise ValueError("Maximum 50 text queries allowed")
        
        for query in v:
            if len(query.strip()) < 3:
                raise ValueError("Each query must be at least 3 characters long")
        
        return v

class BatchPredictionRequest(BaseRequest):
    """Request for batch medical image prediction."""
    image_data_list: List[Union[str, bytes]] = Field(description="List of medical images")
    query: Optional[str] = Field(None, description="Clinical query to apply to all images")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    medical_domain: MedicalDomain = Field(default=MedicalDomain.GENERAL, description="Medical domain")
    batch_metadata: Optional[Dict[str, Any]] = Field(None, description="Batch processing metadata")
    
    @validator('image_data_list')
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if not v:
            raise ValueError("At least one image is required")
        
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 images")
        
        return v

class StreamingPredictionRequest(BaseRequest):
    """Request for streaming prediction with priority."""
    image_data: Union[str, bytes] = Field(description="Medical image data")
    query: Optional[str] = Field(None, description="Clinical query")
    priority: Priority = Field(default=Priority.NORMAL, description="Request priority level")
    timeout_ms: float = Field(default=5000.0, description="Request timeout in milliseconds")
    callback_url: Optional[str] = Field(None, description="Optional webhook URL for result notification")

# Response schemas

class PredictionResponse(BaseResponse):
    """Response for medical image prediction."""
    prediction: str = Field(description="Prediction result (positive/negative/uncertain)")
    confidence: float = Field(description="Confidence score (0-1)")
    similarity_score: float = Field(description="Raw similarity score")
    medical_domain: str = Field(description="Medical domain used")
    query_used: str = Field(description="Clinical query that was used")
    model_info: Dict[str, Any] = Field(description="Model information")
    clinical_notes: Optional[str] = Field(None, description="Additional clinical observations")

class SimilarityMatch(BaseModel):
    """Single similarity match result."""
    query: str = Field(description="Text query")
    similarity_score: float = Field(description="Similarity score")
    confidence: float = Field(description="Confidence in the match")
    rank: Optional[int] = Field(None, description="Rank in results")

class SimilarityResponse(BaseResponse):
    """Response for similarity computation."""
    similarities: List[SimilarityMatch] = Field(description="List of similarity results")
    best_match: SimilarityMatch = Field(description="Best matching query")
    model_info: Dict[str, Any] = Field(description="Model information")

class BatchPredictionResult(BaseModel):
    """Single result in batch prediction."""
    image_index: int = Field(description="Index of image in batch")
    prediction: str = Field(description="Prediction result")
    confidence: float = Field(description="Confidence score")
    similarity_score: float = Field(description="Raw similarity score")
    status: PredictionStatus = Field(description="Processing status")
    error: Optional[str] = Field(None, description="Error message if failed")

class BatchStatistics(BaseModel):
    """Statistics for batch processing."""
    total_images: int = Field(description="Total number of images processed")
    successful_predictions: int = Field(description="Number of successful predictions")
    failed_predictions: int = Field(description="Number of failed predictions")
    success_rate: float = Field(description="Success rate (0-1)")
    average_confidence: float = Field(description="Average confidence of successful predictions")
    processing_time_per_image_ms: Optional[float] = Field(None, description="Average processing time per image")

class BatchPredictionResponse(BaseResponse):
    """Response for batch prediction."""
    results: List[BatchPredictionResult] = Field(description="Individual prediction results")
    batch_statistics: BatchStatistics = Field(description="Batch processing statistics")
    query_used: str = Field(description="Clinical query used for all images")

class StreamingResponse(BaseResponse):
    """Response for streaming prediction."""
    request_id: str = Field(description="Unique request identifier")
    status: PredictionStatus = Field(description="Current request status")
    prediction: Optional[str] = Field(None, description="Prediction result if complete")
    confidence: Optional[float] = Field(None, description="Confidence score if complete")
    similarity_score: Optional[float] = Field(None, description="Similarity score if complete")
    queue_time_ms: Optional[float] = Field(None, description="Time spent in queue")
    error_message: Optional[str] = Field(None, description="Error message if failed")

# Health and monitoring schemas

class ModelStatus(BaseModel):
    """Status of individual model backend."""
    backend: str = Field(description="Backend name")
    status: str = Field(description="Health status")
    last_inference_time: Optional[float] = Field(None, description="Last successful inference timestamp")
    error_count: int = Field(default=0, description="Number of recent errors")
    average_latency_ms: Optional[float] = Field(None, description="Average inference latency")

class HealthResponse(BaseModel):
    """Application health check response."""
    status: str = Field(description="Overall health status")
    timestamp: float = Field(description="Health check timestamp")
    models: Dict[str, str] = Field(description="Model backend statuses")
    streaming_engine: str = Field(description="Streaming engine status")
    metrics: Dict[str, Any] = Field(description="Application metrics")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

class MetricsResponse(BaseModel):
    """Application metrics response."""
    application_metrics: Dict[str, Any] = Field(description="Application-level metrics")
    streaming_metrics: Dict[str, Any] = Field(description="Streaming engine metrics")
    model_info: Dict[str, Any] = Field(description="Model information")
    system_info: Dict[str, Any] = Field(description="System information")
    timestamp: float = Field(description="Metrics collection timestamp")

class ModelInfo(BaseModel):
    """Model information schema."""
    backend: str = Field(description="Backend name")
    model_type: str = Field(description="Model type")
    optimization_level: str = Field(description="Optimization level")
    memory_usage_mb: Union[float, str] = Field(description="Memory usage in MB")
    average_latency_ms: Union[float, str] = Field(description="Average latency in milliseconds")
    supported_features: List[str] = Field(description="Supported features")
    medical_domains: List[str] = Field(description="Supported medical domains")

# Error schemas

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(description="Error message")
    status_code: int = Field(description="HTTP status code")
    timestamp: float = Field(description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID if available")

class ValidationError(BaseModel):
    """Validation error details."""
    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="Invalid value that was provided")

class ValidationErrorResponse(ErrorResponse):
    """Response for validation errors."""
    validation_errors: List[ValidationError] = Field(description="List of validation errors")

# Webhook schemas

class WebhookNotification(BaseModel):
    """Webhook notification for completed requests."""
    request_id: str = Field(description="Request ID")
    status: PredictionStatus = Field(description="Final status")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data if successful")
    timestamp: float = Field(description="Completion timestamp")
    processing_summary: Dict[str, Any] = Field(description="Processing summary")

# Configuration schemas

class APIConfiguration(BaseModel):
    """API configuration schema."""
    max_batch_size: int = Field(default=100, description="Maximum batch size")
    default_confidence_threshold: float = Field(default=0.5, description="Default confidence threshold")
    supported_backends: List[BackendType] = Field(description="Supported model backends")
    rate_limits: Dict[str, int] = Field(description="Rate limits by endpoint")
    authentication_required: bool = Field(default=True, description="Whether authentication is required")

# Example data for documentation

class ExampleData:
    """Example data for API documentation."""
    
    SAMPLE_BASE64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    SAMPLE_PREDICTION_REQUEST = {
        "image_data": SAMPLE_BASE64_IMAGE,
        "query": "chest x-ray showing pneumonia",
        "medical_domain": "radiology",
        "confidence_threshold": 0.7,
        "backend": "auto"
    }
    
    SAMPLE_SIMILARITY_REQUEST = {
        "image_data": SAMPLE_BASE64_IMAGE,
        "text_queries": [
            "normal chest x-ray",
            "pneumonia infiltrates", 
            "pleural effusion",
            "cardiomegaly"
        ],
        "backend": "tensorrt"
    }
    
    SAMPLE_BATCH_REQUEST = {
        "image_data_list": [SAMPLE_BASE64_IMAGE, SAMPLE_BASE64_IMAGE],
        "query": "skin lesion melanoma",
        "medical_domain": "dermatology",
        "confidence_threshold": 0.8,
        "backend": "onnx"
    }

# Add examples to schema configs
PredictionRequest.Config.schema_extra = {
    "example": ExampleData.SAMPLE_PREDICTION_REQUEST
}

SimilarityRequest.Config.schema_extra = {
    "example": ExampleData.SAMPLE_SIMILARITY_REQUEST
}

BatchPredictionRequest.Config.schema_extra = {
    "example": ExampleData.SAMPLE_BATCH_REQUEST
}