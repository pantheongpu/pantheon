# PANTHEON Build System

# --- Compiler Configuration ---
HIPCC ?= hipcc
CXXFLAGS := -O3 -std=c++14 -DNDEBUG -Ikernels/common
BUILD_DIR := build
ROCM_PATH ?= /opt/rocm
BUILD_STAMP := $(BUILD_DIR)/.platform

# Platform Detection
ifeq ($(PLATFORM), MOCK)
    COMPILER := g++
    # Enable Mock Flag and suppress warnings about GPU-specific pragmas
    CXXFLAGS += -DPANTHEON_MOCK -Wno-unknown-pragmas -pthread

else ifeq ($(PLATFORM), CUDA)
    COMPILER := nvcc
    
    # 1. Detect physical GPU architecture
    # nvidia-smi writes "NVIDIA-SMI has failed because it couldn't communicate
    # with the NVIDIA driver" to STDOUT, so 2>/dev/null does not suppress it.
    # Without the numeric filter that sentence becomes DETECTED_ARCH: it is
    # non-empty, so the failsafe below never fires, and it reaches the compiler
    # as sm_<sentence> -- whose apostrophe breaks shell quoting, producing
    # "/bin/sh: Syntax error: Unterminated quoted string" instead of the
    # readable fallback this block was written to provide.
    DETECTED_ARCH := $(strip $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.' | grep -E '^[0-9]+$$'))
    # Print it immediately during the parsing phase
    $(info [DEBUG] The detected architecture is: $(DETECTED_ARCH))

    # 2. Hopper WGMMA Promotion
    # Hopper requires the 'a' suffix (sm_90a) to unlock WGMMA PTX instructions
    ifeq ($(DETECTED_ARCH), 90)
        DETECTED_ARCH := 90a
    endif

    # 3. The Ultimate Failsafe
    # If nvidia-smi fails (e.g., compiling on a CPU-only login node), 
    # default to Hopper (90a) so the build doesn't crash with an empty 'sm_' flag.
    ifeq ($(DETECTED_ARCH),)
        $(info [DEBUG] Hardware detection failed. Falling back to default sm_86...)
        DETECTED_ARCH := 86 
    endif

    # --- Vendored Headers ---
    # OptiX headers are NVIDIA-proprietary and are not redistributable, so this
    # tree does not require them. Set OPTIX_PATH to an OptiX SDK include
    # directory to build the real RT kernel; otherwise rt_virus builds its dummy
    # kernel, exactly as it already does when HIP-RT is missing on AMD.
    OPTIX_PATH ?= $(wildcard kernels/common/optix)
    ifneq ($(OPTIX_PATH),)
        OPTIX_INC := -I $(OPTIX_PATH)
    else
        OPTIX_INC :=
        $(info [DEBUG] OptiX headers not found; rt_virus will build its dummy kernel. Set OPTIX_PATH to enable it.)
    endif
    NVENC_INC := -I kernels/common/nvenc

    CXXFLAGS += -x cu --gpu-architecture=sm_$(DETECTED_ARCH) -Wno-deprecated-gpu-targets $(OPTIX_INC) $(NVENC_INC)
    
    # --- Link the CUDA Driver API and Dynamic Loader ---
    LDFLAGS += -lcuda -ldl

else
    COMPILER := $(HIPCC)
    
    # 1. Detect physical GPU architecture (Modern ROCm uses amdgpu-arch)
    DETECTED_GFX := $(strip $(shell amdgpu-arch 2>/dev/null | head -n 1))
    ifeq ($(DETECTED_GFX),)
        DETECTED_GFX := $(strip $(shell rocm_agent_enumerator 2>/dev/null | grep -v "cpu" | head -n 1))
    endif
    
    # 2. Manual Override & Failsafe
    # Allows cross-compiling on CPU nodes via: make PLATFORM=HIP TARGET_GFX=gfx950
    ifdef TARGET_GFX
        DETECTED_GFX := $(TARGET_GFX)
    endif

    # Tool output can include feature suffixes such as gfx90a:sramecc+:xnack-.
    # hipcc's --offload-arch expects the architecture name only.
    DETECTED_GFX := $(firstword $(subst :, ,$(DETECTED_GFX)))
    
    ifeq ($(DETECTED_GFX),)
        $(info [DEBUG] Hardware detection failed. Falling back to MI300X default...)
        DETECTED_GFX := gfx942
    endif
    
    $(info [DEBUG] The AMD target architecture is: $(DETECTED_GFX))

    # RDNA-family parts (gfx11xx, gfx12xx) provide WMMA; CDNA Instinct parts
    # provide MFMA. Match on the two-digit family prefix rather than on
    # individual model numbers, so any member of a family -- including ones
    # that do not exist yet -- selects the right path with no source change.
    GFX_FAMILY := $(shell printf '%s' '$(DETECTED_GFX)' | sed -E 's/^(gfx[0-9]{2}).*/\1/')
    ifneq ($(filter gfx11 gfx12,$(GFX_FAMILY)),)
        CXXFLAGS += -DPANTHEON_AMD_WMMA_TARGET=1
        $(info [DEBUG] WMMA-capable family detected ($(GFX_FAMILY)); enabling WMMA path.)
    endif
  
    # 3. FIX: Force Native Compilation for Pantheon Kernels
    # If a specific cross-compile target wasn't requested, trust the compiler to 
    # generate machine code for the exact silicon it is currently sitting on.
    # This completely prevents Code 98 on bleeding-edge chips like the MI350X.
    ifdef TARGET_GFX
	CXXFLAGS += -std=c++17 --offload-arch=$(DETECTED_GFX)
    else
        CXXFLAGS += -std=c++17 --offload-arch=native
    endif
    
    # --- Optional Local HIP-RT Vendoring ---
    # The RT kernel builds a dummy fallback when HIP-RT headers are absent, so
    # do not make every AMD build clone and compile HIP-RT by default.
    VENDOR_DIR := vendor
    HIPRT_DIR := $(VENDOR_DIR)/HIPRT
    HIPRT_INSTALL := $(VENDOR_DIR)/install
    HIPRT_TARGET := $(HIPRT_INSTALL)/lib/libhiprt64.so

    ifeq ($(ENABLE_HIPRT),1)
        CXXFLAGS += -I$(abspath $(HIPRT_INSTALL)/include)
        LDFLAGS += -L$(abspath $(HIPRT_INSTALL)/lib) -lhiprt64 -Wl,-rpath=$(abspath $(HIPRT_INSTALL)/lib)
        PLATFORM_DEPS := $(HIPRT_TARGET)
    endif
endif

# OPTIX_PATH is part of the signature: setting it after a build without it
# must invalidate the cached binaries, or rt_virus silently keeps running the
# dummy kernel it compiled when the headers were absent.
BUILD_SIGNATURE := PLATFORM=$(PLATFORM);CUDA_ARCH=$(DETECTED_ARCH);AMD_GFX=$(DETECTED_GFX);TARGET_GFX=$(DETECTED_GFX);ENABLE_HIPRT=$(ENABLE_HIPRT);OPTIX=$(OPTIX_PATH)

# --- Auto-Discovery Logic ---

# 1. Find all .cpp files inside kernels/ subdirectories
ALL_SRCS := $(shell find kernels -name "*.cpp")

# 2. Exclude anything in the 'common' directory or root headers
SRCS := $(filter-out kernels/common/%, $(ALL_SRCS))

# Empty discovery means find is unavailable (minimal container images such as
# rockylinux:8 omit findutils) or the kernels tree is missing. Without this
# guard make exits 0 having built nothing and every workload fails at launch.
ifeq ($(strip $(SRCS)),)
    $(error No kernel sources found under kernels/. Verify that 'find' is installed and the kernels/ directory exists)
endif

# 3. Determine Target Binaries
BINS := $(foreach src,$(SRCS),$(BUILD_DIR)/$(basename $(notdir $(src))))

# Shared headers included by many kernel translation units.
COMMON_HEADERS := kernels/common/common.h kernels/common/ai_workload_template.h

# --- Targets ---

.PHONY: all clean directories FORCE

# Add PLATFORM_DEPS here so HIP-RT is built before the binaries
all: directories $(BUILD_STAMP) $(PLATFORM_DEPS) $(BINS)

$(BUILD_STAMP): FORCE | directories
	@if [ ! -f "$@" ] || [ "$$(cat "$@")" != "$(BUILD_SIGNATURE)" ]; then \
		echo "[BUILD] Platform signature changed. Clearing stale binaries..."; \
		find $(BUILD_DIR) -maxdepth 1 -type f ! -name ".platform" -delete; \
		printf '%s\n' '$(BUILD_SIGNATURE)' > "$@"; \
	fi

$(BINS): $(BUILD_STAMP)

# --- Auto-Dependency: HIP-RT ---
$(HIPRT_TARGET):
	@echo "============================================================"
	@echo "[DEPENDENCY] Fetching and building HIP-RT locally..."
	@echo "============================================================"
	@mkdir -p $(VENDOR_DIR)
	@if [ ! -d "$(HIPRT_DIR)" ]; then \
		git clone https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git $(HIPRT_DIR); \
		cd $(HIPRT_DIR) && git submodule update --init --recursive; \
	fi
	@echo "[DEPENDENCY] Injecting $(DETECTED_GFX) architecture into AMD build scripts..."
	@find $(HIPRT_DIR) -type f \( -name "*.py" -o -name "CMakeLists.txt" \) -exec sed -i "s/'gfx942'/'gfx942', '$(DETECTED_GFX)'/g" {} +
	@find $(HIPRT_DIR) -type f \( -name "*.py" -o -name "CMakeLists.txt" \) -exec sed -i 's/"gfx942"/"gfx942", "$(DETECTED_GFX)"/g' {} +
	@find $(HIPRT_DIR) -type f -name "CMakeLists.txt" -exec sed -i 's/gfx942/gfx942;$(DETECTED_GFX)/g' {} +
	@mkdir -p $(HIPRT_DIR)/build
	@cd $(HIPRT_DIR)/build && cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=ON -DPRECOMPILE=ON -DFORCE_DISABLE_CUDA=ON -DHIP_PATH=$(ROCM_PATH) -DPYTHON_EXECUTABLE=/usr/bin/python3 -DAMDGPU_TARGETS=$(DETECTED_GFX) -DGPU_TARGETS=$(DETECTED_GFX) ..
	@cd $(HIPRT_DIR)/build && cmake --build . --config Release -j $$(nproc)
	@echo "[DEPENDENCY] Build finished. Extracting all runtime assets..."
	@mkdir -p $(HIPRT_INSTALL)/lib
	@mkdir -p $(HIPRT_INSTALL)/include
	@find $(HIPRT_DIR) -type f \( -name "*.hipfb" -o -name "*.bc" -o -name "libhiprt*64.so*" \) -exec cp -L {} $(HIPRT_INSTALL)/lib/ \;
	@REAL_SO=$$(find $(HIPRT_INSTALL)/lib -name "libhiprt*64.so*" -type f | head -n 1); \
	 ln -sf $$(basename $$REAL_SO) $(HIPRT_INSTALL)/lib/libhiprt64.so
	@cp -r $(HIPRT_DIR)/hiprt $(HIPRT_INSTALL)/include/
	@echo "[DEPENDENCY] HIP-RT and Orochi assets successfully installed to local vendor folder!"

# Create build directory
directories:
	@mkdir -p $(BUILD_DIR)

# Generic Rule: Build any binary from its corresponding source found in kernels tree
vpath %.cpp $(sort $(dir $(SRCS)))

$(BUILD_DIR)/%: %.cpp $(COMMON_HEADERS) $(BUILD_STAMP)
	@echo "[BUILD] Compiling $< -> $@"
	@$(COMPILER) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@echo "[CLEAN] Removing build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(VENDOR_DIR)
