#!/usr/bin/env python3
"""
Healthcare VLM API Health Check Script

This script performs comprehensive health checks for the containerized healthcare API.
Designed for use with Docker HEALTHCHECK and container orchestration systems.

Health Checks:
- API endpoint availability
- Model loading capability
- GPU/CUDA availability
- Memory usage validation
- Critical dependency verification
"""

import sys
import time
import requests
import json
import logging
from pathlib import Path
import subprocess
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health check system for healthcare VLM API."""
    
    def __init__(self, api_host="localhost", api_port=8000):
        self.api_host = api_host
        self.api_port = api_port
        self.base_url = f"http://{api_host}:{api_port}"
        self.timeout = 10  # seconds
        
    def check_api_endpoint(self) -> bool:
        """Check if API health endpoint is responding."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info("✓ API health endpoint responding correctly")
                    return True
                else:
                    logger.error(f"✗ API reports unhealthy status: {health_data}")
                    return False
            else:
                logger.error(f"✗ API health endpoint returned status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to reach API health endpoint: {e}")
            return False
    
    def check_python_dependencies(self) -> bool:
        """Check critical Python dependencies."""
        critical_deps = [
            "fastapi",
            "uvicorn", 
            "torch",
            "transformers",
            "numpy",
            "pillow"
        ]
        
        optional_deps = [
            "tensorrt",
            "onnxruntime",
            "open_clip_torch"
        ]
        
        missing_critical = []
        missing_optional = []
        
        for dep in critical_deps:
            try:
                __import__(dep)
                logger.info(f"✓ {dep} available")
            except ImportError:
                missing_critical.append(dep)
                logger.error(f"✗ Critical dependency missing: {dep}")
        
        for dep in optional_deps:
            try:
                __import__(dep)
                logger.info(f"✓ {dep} available (optional)")
            except ImportError:
                missing_optional.append(dep)
                logger.warning(f"! Optional dependency missing: {dep}")
        
        if missing_critical:
            logger.error(f"Critical dependencies missing: {missing_critical}")
            return False
        
        if missing_optional:
            logger.info(f"Optional dependencies missing: {missing_optional}")
        
        return True
    
    def check_cuda_availability(self) -> bool:
        """Check CUDA and GPU availability."""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3   # GB
                
                logger.info(f"✓ CUDA available with {device_count} device(s)")
                logger.info(f"  Current device: {current_device} ({device_name})")
                logger.info(f"  Memory allocated: {memory_allocated:.2f} GB")
                logger.info(f"  Memory reserved: {memory_reserved:.2f} GB")
                
                return True
            else:
                logger.warning("! CUDA not available, running in CPU mode")
                return True  # Not a failure for health check
                
        except Exception as e:
            logger.error(f"✗ Error checking CUDA availability: {e}")
            return False
    
    def check_system_resources(self) -> bool:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / 1024**3
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.used / disk.total * 100
            disk_free_gb = disk.free / 1024**3
            
            logger.info(f"System resources:")
            logger.info(f"  CPU usage: {cpu_percent}%")
            logger.info(f"  Memory usage: {memory_percent}% (Available: {memory_available_gb:.1f} GB)")
            logger.info(f"  Disk usage: {disk_percent:.1f}% (Free: {disk_free_gb:.1f} GB)")
            
            # Health thresholds
            if memory_percent > 90:
                logger.warning(f"! High memory usage: {memory_percent}%")
            
            if disk_percent > 95:
                logger.error(f"✗ Critical disk usage: {disk_percent}%")
                return False
            
            if memory_available_gb < 1.0:
                logger.error(f"✗ Low available memory: {memory_available_gb:.1f} GB")
                return False
            
            logger.info("✓ System resources within acceptable limits")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error checking system resources: {e}")
            return False
    
    def check_model_loading(self) -> bool:
        """Test basic model loading capability."""
        try:
            # Add app directory to path
            sys.path.insert(0, '/app')
            
            # Test model loader import
            from src.models.load_biomedclip import BiomedCLIPLoader
            
            # Quick instantiation test (don't actually load model)
            loader = BiomedCLIPLoader(device='cpu')  # Use CPU for health check
            
            logger.info("✓ Model loading capability verified")
            return True
            
        except ImportError as e:
            logger.error(f"✗ Model loading import error: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Model loading test error: {e}")
            return False
    
    def check_file_permissions(self) -> bool:
        """Check file and directory permissions."""
        required_dirs = [
            "/app/logs",
            "/app/cache", 
            "/app/models"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            
            # Check if directory exists
            if not path.exists():
                logger.error(f"✗ Required directory missing: {dir_path}")
                return False
            
            # Check if writable
            if not path.is_dir():
                logger.error(f"✗ Path is not a directory: {dir_path}")
                return False
            
            # Test write access
            test_file = path / "health_check_test.tmp"
            try:
                test_file.touch()
                test_file.unlink()
                logger.info(f"✓ Directory writable: {dir_path}")
            except PermissionError:
                logger.error(f"✗ Directory not writable: {dir_path}")
                return False
            except Exception as e:
                logger.error(f"✗ Error testing directory {dir_path}: {e}")
                return False
        
        return True
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity for model downloads."""
        test_urls = [
            "https://huggingface.co",
            "https://github.com"
        ]
        
        for url in test_urls:
            try:
                response = requests.head(url, timeout=5)
                if response.status_code < 400:
                    logger.info(f"✓ Network connectivity to {url}")
                else:
                    logger.warning(f"! Network issue with {url}: {response.status_code}")
            except Exception as e:
                logger.warning(f"! Network connectivity issue with {url}: {e}")
        
        # Network connectivity is not critical for container health
        return True
    
    def run_comprehensive_health_check(self) -> bool:
        """Run all health checks and return overall status."""
        logger.info("Starting comprehensive health check...")
        
        checks = [
            ("Python Dependencies", self.check_python_dependencies),
            ("CUDA Availability", self.check_cuda_availability),
            ("System Resources", self.check_system_resources), 
            ("File Permissions", self.check_file_permissions),
            ("Model Loading", self.check_model_loading),
            ("Network Connectivity", self.check_network_connectivity),
            ("API Endpoint", self.check_api_endpoint),
        ]
        
        results = {}
        overall_healthy = True
        
        for check_name, check_func in checks:
            logger.info(f"\n--- Running {check_name} Check ---")
            try:
                result = check_func()
                results[check_name] = result
                
                if not result:
                    overall_healthy = False
                    logger.error(f"✗ {check_name} check failed")
                else:
                    logger.info(f"✓ {check_name} check passed")
                    
            except Exception as e:
                logger.error(f"✗ {check_name} check encountered error: {e}")
                results[check_name] = False
                overall_healthy = False
        
        # Summary
        logger.info("\n--- Health Check Summary ---")
        for check_name, result in results.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"{check_name}: {status}")
        
        overall_status = "HEALTHY" if overall_healthy else "UNHEALTHY"
        logger.info(f"Overall Status: {overall_status}")
        
        return overall_healthy

def main():
    """Main health check execution."""
    try:
        # Create health checker
        checker = HealthChecker()
        
        # Run health check
        is_healthy = checker.run_comprehensive_health_check()
        
        # Exit with appropriate code
        if is_healthy:
            logger.info("✓ All health checks passed")
            sys.exit(0)
        else:
            logger.error("✗ Health check failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Health check script error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()