# Makefile for GPT-2 Attention Optimization Project

# Directories
PROFILE_DIR = results/profiles
METRICS_DIR = results/metrics
BENCHMARK_DIR = benchmarks

# Python interpreter
PYTHON = python

# =============================================================================
# BUILD TARGETS
# =============================================================================

.PHONY: build clean install rebuild

build:
	@$(PYTHON) setup.py build_ext --inplace

clean:
	@$(PYTHON) setup.py clean --all
	@rm -rf build dist *.egg-info
	@find . -name "*.so" -delete
	@find . -name "__pycache__" -delete

install: build
	@$(PYTHON) setup.py install

rebuild: clean build

# =============================================================================
# TESTING TARGETS
# =============================================================================

.PHONY: test test-naive test-tiled test-all

test: test-naive test-tiled

test-naive:
	@$(PYTHON) tests/test_naive_kernels.py

test-tiled:
	@$(PYTHON) tests/test_tiled_qk.py

test-all:
	@$(PYTHON) tests/test_attention.py
	@$(PYTHON) tests/test_naive_kernels.py
	@$(PYTHON) tests/test_tiled_qk.py

# =============================================================================
# BENCHMARKING TARGETS
# =============================================================================

.PHONY: benchmark benchmark-naive benchmark-pytorch benchmark-tiled benchmark-compare

benchmark: benchmark-compare

benchmark-naive:
	@$(PYTHON) $(BENCHMARK_DIR)/profile_naive.py

benchmark-pytorch:
	@$(PYTHON) $(BENCHMARK_DIR)/profile_pytorch.py

benchmark-tiled:
	@$(PYTHON) $(BENCHMARK_DIR)/compare_naive_vs_tiled.py

benchmark-compare: benchmark-naive benchmark-pytorch
	@$(PYTHON) $(BENCHMARK_DIR)/analyze_results.py

# =============================================================================
# NSIGHT SYSTEMS PROFILING
# =============================================================================

.PHONY: nsys-naive nsys-all

$(PROFILE_DIR):
	@mkdir -p $(PROFILE_DIR)

nsys-naive: $(PROFILE_DIR)
	@nsys profile \
		--trace=cuda,nvtx \
		--output=$(PROFILE_DIR)/naive_timeline \
		--force-overwrite=true \
		$(PYTHON) $(BENCHMARK_DIR)/profile_naive.py

nsys-all: nsys-naive

# =============================================================================
# NSIGHT COMPUTE PROFILING - NAIVE KERNELS
# =============================================================================

.PHONY: ncu-naive-qk ncu-naive-softmax ncu-naive-av ncu-naive-all

ncu-naive-qk: $(PROFILE_DIR)
	@ncu --set detailed \
		--kernel-name regex:".*naive.*qk.*" \
		--export=$(PROFILE_DIR)/naive_qk \
		--force-overwrite \
		$(PYTHON) $(BENCHMARK_DIR)/profile_naive.py

ncu-naive-softmax: $(PROFILE_DIR)
	@ncu --set detailed \
		--kernel-name regex:".*softmax.*" \
		--export=$(PROFILE_DIR)/naive_softmax \
		--force-overwrite \
		$(PYTHON) $(BENCHMARK_DIR)/profile_naive.py

ncu-naive-av: $(PROFILE_DIR)
	@ncu --set detailed \
		--kernel-name regex:".*naive.*av.*" \
		--export=$(PROFILE_DIR)/naive_av \
		--force-overwrite \
		$(PYTHON) $(BENCHMARK_DIR)/profile_naive.py

ncu-naive-all: ncu-naive-qk ncu-naive-softmax ncu-naive-av

# =============================================================================
# NSIGHT COMPUTE PROFILING - TILED KERNELS
# =============================================================================

.PHONY: ncu-tiled-qk ncu-tiled-av ncu-tiled-all

ncu-tiled-qk: $(PROFILE_DIR)
	@ncu --set detailed \
		--kernel-name regex:".*tiled.*qk.*" \
		--export=$(PROFILE_DIR)/tiled_qk \
		--force-overwrite \
		$(PYTHON) $(BENCHMARK_DIR)/profile_tiled_qk.py

ncu-tiled-av: $(PROFILE_DIR)
	@ncu --set detailed \
		--kernel-name regex:".*tiled.*av.*" \
		--export=$(PROFILE_DIR)/tiled_av \
		--force-overwrite \
		$(PYTHON) $(BENCHMARK_DIR)/profile_tiled_av.py

ncu-tiled-all: ncu-tiled-qk ncu-tiled-av

# =============================================================================
# COMBINED PROFILING TARGETS
# =============================================================================

.PHONY: profile-all profile-quick profile-compare

profile-quick: ncu-naive-qk ncu-tiled-qk

profile-all: nsys-all ncu-naive-all ncu-tiled-all

profile-compare: ncu-naive-qk ncu-tiled-qk
	@ncu-ui $(PROFILE_DIR)/naive_qk.ncu-rep $(PROFILE_DIR)/tiled_qk.ncu-rep

# =============================================================================
# UTILITY TARGETS
# =============================================================================

.PHONY: help clean-profiles show-profiles

clean-profiles:
	@rm -rf $(PROFILE_DIR)/*.ncu-rep $(PROFILE_DIR)/*.nsys-rep

show-profiles:
	@ls -lh $(PROFILE_DIR)/*.ncu-rep $(PROFILE_DIR)/*.nsys-rep 2>/dev/null || echo "No profiles found"
