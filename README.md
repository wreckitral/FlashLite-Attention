# FlashLite-Attention

**FlashLite-Attention** is an educational and research-oriented implementation of the Self-Attention mechanism, inspired by FlashAttention. It is specifically optimized for consumer-grade GPUs with limited VRAM (such as the NVIDIA RTX 3050 4GB).

This project was developed as part of an undergraduate thesis (skripsi) at Universitas Sriwijaya, focusing on memory-efficient CUDA kernels for self-attention in Large Language Model (LLM) inference.

## What is this codebase?

The core purpose of this codebase is to demonstrate and implement **memory-aware optimizations** for the attention mechanism. Standard attention implementations require storing a large $N \times N$ matrix (where $N$ is sequence length), which quickly fills up GPU memory.

This project implements **Kernel Fusion** and **Tiling** techniques to compute attention without materializing the full matrix in global memory, drastically reducing memory usage and improving speed on hardware that is typically considered too limited for large-scale LLM operations.

I implement three progressive kernel versions to demonstrate different optimization techniques:

| Version | Description | Kernels |
|---------|-------------|---------|
| **P0** | Naive baseline | 3 separate kernels (QK, Softmax, AV) |
| **P1** | Tmax | Tiled computation with fused softmaxiled + Online Soft |
| **P2** | FlashLite Fused | Fully fused single kernel |

## Key Achievements

- **92.6% Memory Reduction**: Compared to PyTorch baseline for large sequences.
- **3.44x Speedup**: Over the naive implementation.
- Enables inference with much longer sequences on limited VRAM hardware.

---

## Requirements

- **Hardware**: NVIDIA GPU with CUDA support (RTX 3050 4GB recommended)
- **Software**:
  - CUDA Toolkit 12.x
  - Python 3.8+
  - PyTorch with CUDA support
  - NVIDIA Nsight Compute (for profiling, optional)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FlashLite-Attention
   ```

2. **Install Python dependencies**
   ```bash
   pip install torch numpy pandas plotly streamlit
   ```

3. **Build the CUDA extension**
   ```bash
   python setup.py build_ext --inplace
   ```

## Quick Start

### Running Tests

```bash
# Test P0 (Naive) kernels
python tests/test_p0_correctness.py

# Test P1 (Tiled) kernels
python tests/test_p1_correctness.py

# Test P2 (FlashLite) kernels
python tests/test_p2_correctness.py
```

### Running Benchmarks

```bash
# Benchmark all kernels (generates performance data)
python benchmarks/benchmark_all_kernels.py
```

### Generating Dashboard Data

```bash
# Generate CSV files for dashboard (from benchmark results)
python analyze_all_results.py
```

### Running the Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`

## Advanced Usage

### Using Different Data Directories

You can run kernel benchmarks and generate CSV files to a different directory without affecting your presentation data:

```bash
# 1. Run benchmarks
python benchmarks/benchmark_all_kernels.py

# 2. Generate CSVs to a NEW folder
python analyze_all_results.py -o new_benchmark_data

# 3. Run dashboard with the new data
DASHBOARD_DATA_DIR=new_benchmark_data streamlit run dashboard.py
```

This is useful for:
- Comparing old vs new benchmark results
- Preserving static presentation data in `dashboard_data/`
- Running experiments without modifying the main dashboard

### NVIDIA Nsight Profiling (Optional)

For detailed GPU profiling, you can use NVIDIA Nsight Compute:

```bash
# Profile bottleneck metrics
bash run_bottleneck_profile.sh

# Profile shared memory usage
bash run_shared_memory_profile.sh

# Profile occupancy
bash run_occupancy_profile.sh
```

Then regenerate dashboard data:
```bash
python analyze_all_results.py
```

## Project Structure

```
FlashLite-Attention/
├── benchmarks/              # Benchmark scripts
│   ├── benchmark_all_kernels.py
│   └── ...
├── src/
│   ├── cuda/               # CUDA kernel implementations
│   │   ├── naive_qk.cu
│   │   ├── naive_softmax.cu
│   │   ├── naive_av.cu
│   │   ├── tiled_qk.cu
│   │   ├── tiled_av.cu
│   │   └── flashLite_attention.cu
│   ├── cpp/                # PyTorch C++ bindings
│   └── python/             # Python wrappers
├── tests/                  # Correctness tests
├── dashboard.py            # Streamlit dashboard
├── analyze_all_results.py # Generate CSV/LaTeX tables
├── setup.py               # Build script
└── Makefile               # Build utilities
```

## Running Individual Tests

```bash
# P0: Naive baseline kernels
python tests/test_p0_correctness.py

# P1: Tiled implementation
python tests/test_p1_correctness.py

# P2: Fully fused FlashLite
python tests/test_p2_correctness.py

# Test attention module
python src/python/attention_module.py
```

## Output Files

- **Benchmark Results**: `results/metrics/performance_benchmark.csv`
- **Dashboard CSVs**: `dashboard_data/*.csv`
- **NCU Profiles**: `results/profiles/*.ncu-rep`

## Troubleshooting

### Build Errors

If you encounter build errors:
```bash
# Clean and rebuild
make clean
python setup.py build_ext --inplace
```

### CUDA Errors

- Ensure CUDA toolkit is installed: `nvcc --version`
- Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### Dashboard Not Loading

- Check that CSV files exist in `dashboard_data/`
- Run: `python analyze_all_results.py` to regenerate

## License

This project is for educational purposes (thesis project).

## Author

- **Author**: Defhanaya Sofhiea
