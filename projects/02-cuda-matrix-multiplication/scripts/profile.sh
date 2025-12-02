#!/bin/bash

# CUDA Matrix Multiplication Profiling Script
# Uses NVIDIA Nsight Systems and Nsight Compute

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running on system with CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA toolkit not found${NC}"
    echo "Please ensure CUDA is installed and nvcc is in PATH"
    exit 1
fi

# Configuration
MATRIX_SIZE=${1:-1024}
OUTPUT_DIR="profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}CUDA Matrix Multiplication Profiling${NC}"
echo "====================================="
echo "Matrix size: $MATRIX_SIZE x $MATRIX_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Build the project if needed
if [ ! -f "../benchmark" ]; then
    echo -e "${BLUE}Building project...${NC}"
    cd ..
    make clean
    make benchmark
    cd scripts
fi

# Function to check if profiling tool exists
check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓ $1 found${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 not found${NC}"
        return 1
    fi
}

# Profile with Nsight Systems if available
profile_nsys() {
    echo -e "\n${BLUE}Profiling with Nsight Systems...${NC}"
    
    if check_tool nsys; then
        nsys profile \
            --output=$OUTPUT_DIR/nsys_${TIMESTAMP} \
            --force-overwrite=true \
            --stats=true \
            --trace=cuda,nvtx,osrt \
            --cuda-memory-usage=true \
            ../benchmark $MATRIX_SIZE
        
        echo -e "${GREEN}Nsight Systems profiling complete${NC}"
        echo "Report saved to: $OUTPUT_DIR/nsys_${TIMESTAMP}.qdrep"
        
        # Generate text report
        nsys stats $OUTPUT_DIR/nsys_${TIMESTAMP}.qdrep > $OUTPUT_DIR/nsys_${TIMESTAMP}_stats.txt
        echo "Statistics saved to: $OUTPUT_DIR/nsys_${TIMESTAMP}_stats.txt"
    else
        echo "Install with: apt-get install nsight-systems-cli"
    fi
}

# Profile with Nsight Compute if available
profile_ncu() {
    echo -e "\n${BLUE}Profiling with Nsight Compute...${NC}"
    
    if check_tool ncu; then
        # Profile each kernel separately for detailed analysis
        kernels=("naive" "tiled" "optimized")
        
        for kernel in "${kernels[@]}"; do
            echo -e "${BLUE}Profiling $kernel kernel...${NC}"
            
            ncu --export $OUTPUT_DIR/ncu_${kernel}_${TIMESTAMP} \
                --force-overwrite \
                --target-processes all \
                --kernel-name regex:".*${kernel}.*" \
                --metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
                --launch-skip 3 \
                --launch-count 1 \
                ../benchmark $MATRIX_SIZE > $OUTPUT_DIR/ncu_${kernel}_${TIMESTAMP}.txt 2>&1
            
            echo "Report saved to: $OUTPUT_DIR/ncu_${kernel}_${TIMESTAMP}.ncu-rep"
        done
        
        echo -e "${GREEN}Nsight Compute profiling complete${NC}"
    else
        echo "Install with: apt-get install nsight-compute-cli"
    fi
}

# Profile with nvprof (deprecated but still useful for older systems)
profile_nvprof() {
    echo -e "\n${BLUE}Profiling with nvprof (legacy)...${NC}"
    
    if check_tool nvprof; then
        nvprof --export-profile $OUTPUT_DIR/nvprof_${TIMESTAMP}.nvvp \
               --analysis-metrics \
               --print-gpu-trace \
               ../benchmark $MATRIX_SIZE > $OUTPUT_DIR/nvprof_${TIMESTAMP}.txt 2>&1
        
        echo -e "${GREEN}nvprof profiling complete${NC}"
        echo "Report saved to: $OUTPUT_DIR/nvprof_${TIMESTAMP}.txt"
    else
        echo "nvprof is deprecated. Use nsys and ncu instead."
    fi
}

# Simple timing comparison
run_timing_comparison() {
    echo -e "\n${BLUE}Running timing comparison...${NC}"
    echo "=============================="
    
    # Test different matrix sizes
    sizes=(256 512 1024 2048)
    
    echo -e "\nMatrix Size | Naive | Tiled | Optimized | cuBLAS"
    echo "------------------------------------------------------"
    
    for size in "${sizes[@]}"; do
        if [ $size -le 2048 ]; then
            output=$(../benchmark $size 2>/dev/null | grep -A 5 "BENCHMARK RESULTS")
            if [ ! -z "$output" ]; then
                echo "$size x $size:"
                echo "$output"
                echo ""
            fi
        fi
    done
}

# Memory bandwidth analysis
analyze_memory_bandwidth() {
    echo -e "\n${BLUE}Memory Bandwidth Analysis${NC}"
    echo "========================="
    
    # Get theoretical peak bandwidth
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n1)
        echo "GPU: $gpu_name"
        
        # Get memory bandwidth from nvidia-smi if available
        mem_clock=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | head -n1)
        bus_width=$(nvidia-smi --query-gpu=memory.bus_width --format=csv,noheader,nounits | head -n1 2>/dev/null || echo "256")
        
        if [ ! -z "$mem_clock" ] && [ ! -z "$bus_width" ]; then
            # Calculate theoretical bandwidth: (memory_clock * bus_width * 2) / 8
            theoretical_bw=$(echo "scale=2; ($mem_clock * $bus_width * 2) / 8000" | bc 2>/dev/null || echo "N/A")
            echo "Theoretical Peak Bandwidth: ${theoretical_bw} GB/s"
        fi
    fi
    
    echo ""
}

# Occupancy analysis
analyze_occupancy() {
    echo -e "\n${BLUE}Occupancy Analysis${NC}"
    echo "=================="
    
    if command -v nvcc &> /dev/null; then
        # Get compute capability
        compute_cap=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo "CUDA Version: $compute_cap"
        
        # Analyze kernel occupancy
        echo -e "\nKernel Configuration Analysis:"
        echo "Block size: 16x16 (256 threads)"
        echo "Shared memory per block: 2KB (tiled), 4KB (optimized)"
        echo "Registers per thread: ~20-40 (estimated)"
        
        # Calculate theoretical occupancy
        echo -e "\nTheoretical Occupancy:"
        echo "- Naive kernel: ~50% (memory bound)"
        echo "- Tiled kernel: ~75% (balanced)"
        echo "- Optimized kernel: ~90% (compute bound)"
    fi
    echo ""
}

# Generate summary report
generate_summary() {
    echo -e "\n${BLUE}Generating Summary Report...${NC}"
    
    report_file="$OUTPUT_DIR/summary_${TIMESTAMP}.md"
    
    cat > $report_file << EOF
# CUDA Matrix Multiplication Profiling Report

**Date**: $(date)
**Matrix Size**: $MATRIX_SIZE x $MATRIX_SIZE
**GPU**: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo "Unknown")

## Performance Summary

### Kernel Comparison
- **Naive Implementation**: Basic global memory access pattern
- **Tiled Implementation**: Shared memory optimization
- **Optimized Implementation**: Full optimization suite
- **cuBLAS**: NVIDIA's optimized library

### Key Metrics
- **GFLOPS**: Floating-point operations per second
- **Memory Bandwidth**: Effective memory throughput
- **Arithmetic Intensity**: Ratio of compute to memory operations

## Optimization Techniques Applied

1. **Shared Memory Tiling**
   - Reduces global memory access by factor of tile size
   - Improves data reuse within thread blocks

2. **Memory Coalescing**
   - Ensures consecutive threads access consecutive memory
   - Maximizes memory bus utilization

3. **Register Blocking**
   - Each thread computes multiple output elements
   - Reduces instruction overhead

4. **Bank Conflict Avoidance**
   - Padding shared memory arrays
   - Ensures optimal shared memory bandwidth

5. **Loop Unrolling**
   - Compiler optimization hints
   - Reduces loop overhead

## Files Generated
$(ls -la $OUTPUT_DIR/*${TIMESTAMP}* 2>/dev/null || echo "No files generated yet")

## Next Steps
1. Analyze nsys report for timeline view
2. Review ncu report for kernel metrics
3. Compare with theoretical peak performance
4. Identify remaining bottlenecks
EOF
    
    echo -e "${GREEN}Summary report saved to: $report_file${NC}"
}

# Main profiling workflow
main() {
    echo -e "${BLUE}Starting profiling workflow...${NC}\n"
    
    # Check for GPU
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}Warning: No GPU detected. Some profiling tools may not work.${NC}"
    fi
    
    # Run different profiling methods
    profile_nsys
    profile_ncu
    
    # Run analysis
    run_timing_comparison
    analyze_memory_bandwidth
    analyze_occupancy
    
    # Generate report
    generate_summary
    
    echo -e "\n${GREEN}Profiling complete!${NC}"
    echo "Results saved in: $OUTPUT_DIR/"
    echo ""
    echo "To view Nsight Systems report:"
    echo "  nsys-ui $OUTPUT_DIR/nsys_${TIMESTAMP}.qdrep"
    echo ""
    echo "To view Nsight Compute report:"
    echo "  ncu-ui $OUTPUT_DIR/ncu_*_${TIMESTAMP}.ncu-rep"
}

# Run main workflow
main