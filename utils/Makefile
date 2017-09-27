# Unix commands.
PYTHON := python
NVCC_COMPILE := nvcc -c -o
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared -O2

# File structure.
BUILD_DIR := build
INCLUDE_DIRS := cinclude
TORCH_FFI_BUILD := build_ffi.py
POINTNET2_KERNELS := $(BUILD_DIR)/ball_query_gpu.so $(BUILD_DIR)/group_points_gpu.so $(BUILD_DIR)/sampling_gpu.so $(BUILD_DIR)/interpolate_gpu.so
TORCH_FFI_TARGET := $(BUILD_DIR)/pointnet2/_pointnet2.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(POINTNET2_KERNELS) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BUILD_DIR)/%.so: csrc/%.cu
	@ mkdir -p $(BUILD_DIR)
	# Separate cpp shared library that will be loaded to the extern C ffi
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS)

clean:
	$(RM_RF) $(BUILD_DIR) $(POINTNET2_KERNELS)
