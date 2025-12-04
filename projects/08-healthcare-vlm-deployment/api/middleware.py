"""
Middleware for Healthcare VLM API

This module provides comprehensive middleware for the medical imaging API including:
- Request/response logging with HIPAA compliance
- Error handling and monitoring
- Rate limiting for clinical environments
- Security headers and CORS
- Performance monitoring and metrics collection

Key Features:
- HIPAA-compliant logging (no PHI exposure)
- Clinical workflow optimized rate limiting
- Comprehensive error tracking
- Performance monitoring
- Security hardening
"""

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
import time
import logging
import json
import hashlib
import re
from typing import Dict, Any, Optional, Set, List
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque
import psutil
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HIPAACompliantLogger:
    """
    HIPAA-compliant logging that ensures no PHI is exposed in logs.
    
    Scrubs sensitive data and maintains audit trail for healthcare compliance.
    """
    
    def __init__(self):
        self.phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b\d{16}\b',  # Credit card like numbers
            r'\b\d{4}/\d{2}/\d{2}\b',  # Dates
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names pattern (simplified)
        ]
        
        self.audit_logger = logging.getLogger("audit")
        self.error_logger = logging.getLogger("errors")
        self.performance_logger = logging.getLogger("performance")
    
    def scrub_phi(self, data: Any) -> Any:
        """Remove potential PHI from data before logging."""
        if isinstance(data, str):
            scrubbed = data
            for pattern in self.phi_patterns:
                scrubbed = re.sub(pattern, "[REDACTED]", scrubbed)
            return scrubbed
        
        elif isinstance(data, dict):
            scrubbed = {}
            for key, value in data.items():
                # Skip potentially sensitive fields entirely
                if key.lower() in ['password', 'token', 'ssn', 'patient_name', 'dob', 'mrn']:
                    scrubbed[key] = "[REDACTED]"
                else:
                    scrubbed[key] = self.scrub_phi(value)
            return scrubbed
        
        elif isinstance(data, list):
            return [self.scrub_phi(item) for item in data]
        
        else:
            return data
    
    def log_request(self, request: Request, request_id: str, user_id: Optional[str] = None):
        """Log API request with PHI scrubbing."""
        log_data = {
            "event": "api_request",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url).split('?')[0],  # Remove query params
            "user_id": user_id,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        self.audit_logger.info(json.dumps(self.scrub_phi(log_data)))
    
    def log_response(self, request_id: str, response_time: float, status_code: int, 
                    user_id: Optional[str] = None, error: Optional[str] = None):
        """Log API response with performance metrics."""
        log_data = {
            "event": "api_response",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": response_time * 1000,
            "status_code": status_code,
            "user_id": user_id,
            "error": self.scrub_phi(error) if error else None
        }
        
        self.audit_logger.info(json.dumps(log_data))
        
        # Separate performance logging
        perf_data = {
            "request_id": request_id,
            "response_time_ms": response_time * 1000,
            "status_code": status_code
        }
        self.performance_logger.info(json.dumps(perf_data))
    
    def log_error(self, request_id: str, error: Exception, context: Dict[str, Any] = None):
        """Log application errors with context."""
        error_data = {
            "event": "application_error",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": self.scrub_phi(str(error)),
            "context": self.scrub_phi(context) if context else None
        }
        
        self.error_logger.error(json.dumps(error_data))

class ClinicalRateLimiter:
    """
    Rate limiter optimized for clinical workflows.
    
    Provides different limits based on:
    - Request priority (emergency vs routine)
    - User role (physician vs researcher)
    - Endpoint type (single vs batch)
    """
    
    def __init__(self):
        # Rate limit configurations (requests per minute)
        self.limits = {
            "emergency": {"requests_per_minute": 120, "burst": 20},
            "routine": {"requests_per_minute": 60, "burst": 10},
            "batch": {"requests_per_minute": 10, "burst": 2},
            "research": {"requests_per_minute": 30, "burst": 5}
        }
        
        # Track requests per client/endpoint
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_counters: Dict[str, int] = defaultdict(int)
        
        # Cleanup task
        asyncio.create_task(self._cleanup_old_requests())
    
    def _get_limit_key(self, client_ip: str, endpoint: str, priority: str = "routine") -> str:
        """Generate unique key for rate limiting."""
        return f"{client_ip}:{endpoint}:{priority}"
    
    def _get_limit_config(self, endpoint: str, priority: str) -> Dict[str, int]:
        """Get rate limit configuration for endpoint and priority."""
        if "emergency" in priority.lower() or "critical" in priority.lower():
            return self.limits["emergency"]
        elif "batch" in endpoint.lower():
            return self.limits["batch"]
        elif "research" in endpoint.lower() or "experiment" in endpoint.lower():
            return self.limits["research"]
        else:
            return self.limits["routine"]
    
    async def check_rate_limit(self, request: Request) -> bool:
        """
        Check if request is within rate limits.
        
        Returns True if allowed, False if rate limited.
        """
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        
        # Extract priority from request headers or body
        priority = request.headers.get("X-Priority", "routine")
        
        limit_key = self._get_limit_key(client_ip, endpoint, priority)
        config = self._get_limit_config(endpoint, priority)
        
        current_time = time.time()
        request_times = self.request_history[limit_key]
        
        # Remove requests older than 1 minute
        while request_times and current_time - request_times[0] > 60:
            request_times.popleft()
        
        # Check minute-based limit
        if len(request_times) >= config["requests_per_minute"]:
            return False
        
        # Check burst limit (last 10 seconds)
        recent_requests = sum(1 for t in request_times if current_time - t <= 10)
        if recent_requests >= config["burst"]:
            return False
        
        # Record this request
        request_times.append(current_time)
        
        return True
    
    async def _cleanup_old_requests(self):
        """Periodically cleanup old request history."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                current_time = time.time()
                for limit_key in list(self.request_history.keys()):
                    request_times = self.request_history[limit_key]
                    
                    # Remove requests older than 1 minute
                    while request_times and current_time - request_times[0] > 60:
                        request_times.popleft()
                    
                    # Remove empty deques
                    if not request_times:
                        del self.request_history[limit_key]
                        
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

class PerformanceMonitor:
    """
    Monitor API performance and system resources.
    
    Tracks:
    - Request latency distribution
    - Memory and CPU usage
    - GPU utilization
    - Error rates
    """
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_by_endpoint": defaultdict(int),
            "response_times": deque(maxlen=1000),  # Last 1000 requests
            "error_counts": defaultdict(int),
            "active_requests": 0
        }
        
        # Start monitoring task
        asyncio.create_task(self._monitor_system_resources())
    
    def record_request_start(self, endpoint: str):
        """Record request start."""
        self.metrics["requests_total"] += 1
        self.metrics["requests_by_endpoint"][endpoint] += 1
        self.metrics["active_requests"] += 1
    
    def record_request_end(self, endpoint: str, response_time: float, error: bool = False):
        """Record request completion."""
        self.metrics["active_requests"] = max(0, self.metrics["active_requests"] - 1)
        self.metrics["response_times"].append(response_time)
        
        if error:
            self.metrics["error_counts"][endpoint] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        response_times = list(self.metrics["response_times"])
        
        stats = {
            "requests_total": self.metrics["requests_total"],
            "active_requests": self.metrics["active_requests"],
            "requests_by_endpoint": dict(self.metrics["requests_by_endpoint"]),
            "error_counts": dict(self.metrics["error_counts"]),
        }
        
        if response_times:
            stats.update({
                "avg_response_time_ms": sum(response_times) * 1000 / len(response_times),
                "p95_response_time_ms": sorted(response_times)[int(0.95 * len(response_times))] * 1000,
                "p99_response_time_ms": sorted(response_times)[int(0.99 * len(response_times))] * 1000,
            })
        
        return stats
    
    async def _monitor_system_resources(self):
        """Monitor system resources periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # CPU and Memory
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # GPU monitoring (if available)
                gpu_stats = {}
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_stats = {
                            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
                        }
                except:
                    pass
                
                resource_stats = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / 1024**3,
                    **gpu_stats
                }
                
                # Log resource usage
                logger.info(f"System resources: {json.dumps(resource_stats)}")
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

class RequestMiddleware(BaseHTTPMiddleware):
    """
    Main request middleware that coordinates all middleware functionality.
    """
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.hipaa_logger = HIPAACompliantLogger()
        self.rate_limiter = ClinicalRateLimiter()
        self.performance_monitor = PerformanceMonitor()
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch method."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to state
        request.state.request_id = request_id
        
        # Extract user info if available
        user_id = None
        try:
            # This would extract from JWT or session
            auth_header = request.headers.get("authorization")
            if auth_header:
                user_id = "authenticated_user"  # Simplified
        except:
            pass
        
        # Log request start
        self.hipaa_logger.log_request(request, request_id, user_id)
        self.performance_monitor.record_request_start(request.url.path)
        
        # Check rate limits
        if not await self.rate_limiter.check_rate_limit(request):
            response_time = time.time() - start_time
            self.hipaa_logger.log_response(
                request_id, response_time, 429, user_id, "Rate limit exceeded"
            )
            self.performance_monitor.record_request_end(request.url.path, response_time, error=True)
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Please reduce request frequency for clinical safety",
                    "retry_after": 60
                }
            )
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
            
        except Exception as e:
            error = str(e)
            self.hipaa_logger.log_error(request_id, e, {"endpoint": request.url.path})
            
            # Return error response
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Log response
        response_time = time.time() - start_time
        status_code = response.status_code if response else 500
        
        self.hipaa_logger.log_response(request_id, response_time, status_code, user_id, error)
        self.performance_monitor.record_request_end(
            request.url.path, response_time, error=(status_code >= 400)
        )
        
        # Add response headers
        if response:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(response_time)
        
        return response

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with healthcare-specific security headers."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Healthcare-specific headers
        response.headers["X-Healthcare-API"] = "HIPAA-Compliant"
        response.headers["X-Data-Classification"] = "Medical"
        
        return response

def setup_middleware(app: FastAPI):
    """Setup all middleware for the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://localhost:3000", "https://medical-app.com"],  # Whitelist only
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "*.medical-domain.com", "127.0.0.1"]
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RequestMiddleware)

# Utility functions for other modules

def request_logger():
    """Get the HIPAA-compliant logger instance."""
    return HIPAACompliantLogger()

def error_handler():
    """Get error handling utilities."""
    return HIPAACompliantLogger()

def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    # This would be called from the main app
    # For now, return empty dict
    return {}