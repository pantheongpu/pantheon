#include "../common/common.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <string>

// --- NVIDIA OPTIX GUARD ---
#if defined(__CUDACC__)
    #include <cuda.h>
    // Probed rather than assumed, mirroring the HIP-RT branch below. The OptiX
    // headers are NVIDIA-proprietary and cannot be redistributed, so they are
    // not carried in this tree by default; point OPTIX_PATH at an OptiX SDK to
    // build the real kernel. Without them this degrades to the dummy kernel
    // instead of failing the build.
    #if __has_include(<optix.h>)
        #include <optix.h>
        #include <optix_stubs.h>
        #include <optix_function_table_definition.h>
        #define OPTIX_SUPPORTED 1
    #else
        #define OPTIX_SUPPORTED 0
        #pragma message("OptiX headers not found. Dummy kernel will be built. Set OPTIX_PATH to an OptiX SDK for the real one.")
    #endif
    #define HIPRT_SUPPORTED 0

// --- AMD HIP-RT GUARD ---
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    #define OPTIX_SUPPORTED 0
    #if __has_include(<hiprt/hiprt.h>)
        #include <hiprt/hiprt.h>
        #define HIPRT_SUPPORTED 1
    #else
        #define HIPRT_SUPPORTED 0
        #pragma message("HIP-RT headers not found. Dummy kernel will be built.")
    #endif
#else
    #define OPTIX_SUPPORTED 0
    #define HIPRT_SUPPORTED 0
#endif

// OptiX Error Macro
#if OPTIX_SUPPORTED
#define OPTIX_CHECK( call )                                                    \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::cerr << "[PANTHEON ERROR] OptiX call failed with code " << res << " at line " << __LINE__ << std::endl; \
            exit( 2 );                                                         \
        }                                                                      \
    }
#endif

// --- INDEX INITIALIZATION KERNEL ---
__global__ void init_indices_kernel(uint32_t* indices, size_t num_triangles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Create a 1:1 linear mapping (Triangle 0 gets vertices 0,1,2)
    for (size_t i = idx; i < num_triangles; i += stride) {
        indices[i * 3 + 0] = i * 3 + 0;
        indices[i * 3 + 1] = i * 3 + 1;
        indices[i * 3 + 2] = i * 3 + 2;
    }
}

// --- INITIALIZATION KERNEL ---
__global__ void init_geometry_kernel(float* vertices, size_t num_triangles, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Dynamically scale the geometry spread based on the fuzzer pattern
    float spread = (init_pattern == 0) ? 2.0f : (float)init_pattern * 2.0f;
    float scale = (init_pattern == 0) ? 1.0f : (float)init_pattern;

    for (size_t i = idx; i < num_triangles; i += stride) {
        size_t offset = i * 9; // 3 vertices * 3 floats (x,y,z)
        
        // Spread the triangles out into a 3D grid to prevent overlaps
        // This gives the BVH builder distinct, non-degenerate bounding boxes
        float base_x = (float)(i % 100) * spread;
        float base_y = (float)((i / 100) % 100) * spread;
        float base_z = (float)(i / 10000) * spread;
        
        // Vertex 0 (Origin)
        vertices[offset + 0] = base_x;
        vertices[offset + 1] = base_y;
        vertices[offset + 2] = base_z;
        
        // Vertex 1 (Offset X)
        vertices[offset + 3] = base_x + scale;
        vertices[offset + 4] = base_y;
        vertices[offset + 5] = base_z;
        
        // Vertex 2 (Offset Y)
        vertices[offset + 6] = base_x;
        vertices[offset + 7] = base_y + scale;
        vertices[offset + 8] = base_z;
    }
}

// --- ERROR INJECTION KERNEL ---
__global__ void inject_rt_error_kernel(float* vertices, size_t target_idx) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Intentionally corrupt a vertex coordinate massively to ensure
        // the hardware spatial bounds calculations completely diverge.
        vertices[target_idx] += 99999.0f;
    }
}

// --- VERIFICATION KERNELS ---
__global__ void verify_aabb_kernel(const float* current, const float* golden, unsigned int* err_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Verify the 6 floats (minX, minY, minZ, maxX, maxY, maxZ) of the AABB
        for (int i = 0; i < 6; ++i) {
            if (current[i] != golden[i]) {
                printf("[SDC FAULT][RT_VIRUS] BVH AABB Divergence on Axis %d! Exp: %.2f | Act: %.2f\n", i, golden[i], current[i]);
                atomicAdd(err_count, 1);
            }
        }
    }
}

__global__ void verify_bvh_buffer_kernel(const unsigned int* current, const unsigned int* golden, size_t num_elements, unsigned int* err_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < num_elements; i += stride) {
        if (current[i] != golden[i]) {
            // Cap prints to prevent log spam if the entire buffer shifts
            if (*err_count < 5) {
                printf("[SDC FAULT][RT_VIRUS] BVH Build Divergence! Memory Offset: %llu | Exp: 0x%08x | Act: 0x%08x\n",
                       (unsigned long long)i, golden[i], current[i]);
            }
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 10;     // Number of BVH builds to batch per host synchronization
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 1;      // Controls geometry scaling/spread

    bool verify_mode = false;
    int inject_error = 0;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verify") verify_mode = true;
        if (std::string(argv[i]) == "--inject_error") inject_error = 1;
        if (std::string(argv[i]) == "--block_size" && i+1 < argc) block_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--grid_size" && i+1 < argc) grid_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--kernel_loops" && i+1 < argc) kernel_loops = atoi(argv[++i]);
        if (std::string(argv[i]) == "--warmup_iters" && i+1 < argc) warmup_iters = atoi(argv[++i]);
        if (std::string(argv[i]) == "--sync_mode" && i+1 < argc) sync_mode = atoi(argv[++i]);
        if (std::string(argv[i]) == "--init_pattern" && i+1 < argc) {
            if (!pantheon_parse_init_pattern(argv[++i], &init_pattern)) {
                std::cerr << "[PANTHEON] Unknown --init_pattern '" << argv[i] << "'." << std::endl;
                pantheon_print_init_patterns(std::cerr);
                return 1;
            }
        }
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);

    // --- 1. SET SYNC MODE ---
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__) || defined(__HIP_PLATFORM_HCC__)
    unsigned int sync_flag = hipDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = hipDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = hipDeviceScheduleYield;
    hipSetDeviceFlags(sync_flag);
#elif defined(__CUDACC__)
    unsigned int sync_flag = cudaDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = cudaDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = cudaDeviceScheduleYield;
    cudaSetDeviceFlags(sync_flag);
#else
    // MOCK PLATFORM: CPU execution doesn't require hardware scheduling flags
#endif

    CHECK(hipSetDevice(gpu_id));
    
    hipDeviceProp_t prop; 
    CHECK(hipGetDeviceProperties(&prop, gpu_id));

    // --- HARDWARE SILICON GUARD ---
    // CDNA architectures (MI100, MI200, MI300) are prefixed with 'gfx9'
    // They physically lack the Hardware Ray Accelerators present in RDNA (gfx10/gfx11)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    std::string arch(prop.gcnArchName);
    if (arch.find("gfx9") != std::string::npos) {
        std::cout << "[PANTHEON] GPU " << gpu_id << " Architecture (" << arch << ") is CDNA." << std::endl;
        std::cout << "[PANTHEON] Skipping RT_VIRUS: CDNA silicon physically lacks Hardware Ray Accelerators." << std::endl;
        std::cout << "Throughput: 0.0 GRays/s" << std::endl;
        return 0; // Exit cleanly without failing the suite
    }
#endif

    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        // High occupancy to maximize setup parallelism before handing off to the fixed-function BVH builders
        num_blocks = prop.multiProcessorCount * 16;
        auto_grid = true;
    }

#if OPTIX_SUPPORTED
    // ---------------------------------------------------------
    // NVIDIA OPTIX IMPLEMENTATION
    // ---------------------------------------------------------
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "[PANTHEON ERROR] cuInit(0) failed." << std::endl;
        std::cout << "Throughput: 0.0 GRays/s" << std::endl;
        return 0;
    }

    if (optixInit() != OPTIX_SUCCESS) {
        std::cout << "Throughput: 0.0 GRays/s" << std::endl;
        return 0;
    }

    OptixDeviceContextOptions options = {};
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));

    const int num_triangles = 1000000;
    const int num_vertices = num_triangles * 3;
    size_t num_floats = num_vertices * 3;
    size_t vertices_size = num_floats * sizeof(float);

    float* d_vertices;
    CHECK(hipMalloc(&d_vertices, vertices_size));
    
    hipStream_t stream; CHECK(hipStreamCreate(&stream));

    // Initialize distinct 3D geometry to guarantee BVH determinism
    LAUNCH_KERNEL_ASYNC(init_geometry_kernel, num_blocks, block_size, 0, stream, d_vertices, num_triangles, init_pattern);
    CHECK(hipStreamSynchronize(stream));

    CUdeviceptr d_vertices_ptr = (CUdeviceptr)d_vertices;

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    uint32_t triangle_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
    buildInput.triangleArray.numVertices = num_vertices;
    buildInput.triangleArray.vertexBuffers = &d_vertices_ptr;
    buildInput.triangleArray.flags = triangle_flags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes));

    void *d_temp_buffer, *d_output_buffer;
    CHECK(hipMalloc(&d_temp_buffer, bufferSizes.tempSizeInBytes));
    CHECK(hipMalloc(&d_output_buffer, bufferSizes.outputSizeInBytes));

    float* d_aabb_buffer;
    CHECK(hipMalloc(&d_aabb_buffer, 6 * sizeof(float)));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_AABBS;
    emitDesc.result = (CUdeviceptr)d_aabb_buffer;

    OptixTraversableHandle bvh_handle = 0;

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running RT VIRUS (OptiX BVH Builder)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << " (BVH Builds per Sync)" << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Geometry Scale/Spread)" << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    float* d_golden_aabb = nullptr;
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected OptiX BVH baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_aabb, 6 * sizeof(float)));
        
        OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions, &buildInput, 1,
                                    (CUdeviceptr)d_temp_buffer, bufferSizes.tempSizeInBytes,
                                    (CUdeviceptr)d_output_buffer, bufferSizes.outputSizeInBytes,
                                    &bvh_handle, &emitDesc, 1));
        CHECK(hipStreamSynchronize(stream));
        
        // Snapshot the perfect mathematical bounding box
        CHECK(cudaMemcpy(d_golden_aabb, d_aabb_buffer, 6 * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // --- WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup BVH builds..." << std::endl;
        for (int w = 0; w < warmup_iters; w++) {
            OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions, &buildInput, 1,
                                        (CUdeviceptr)d_temp_buffer, bufferSizes.tempSizeInBytes,
                                        (CUdeviceptr)d_output_buffer, bufferSizes.outputSizeInBytes,
                                        &bvh_handle, &emitDesc, 1));
        }
        CHECK(hipStreamSynchronize(stream));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t builds = 0;
    unsigned int sync_loops = 0;

    // --- ACTIVE LOOP ---
    while(true) {
        // Inject error periodically based on outer sync loop
        if (inject_error && sync_loops == 1) {
            LAUNCH_KERNEL_ASYNC(inject_rt_error_kernel, 1, 1, 0, stream, d_vertices, 1337);
            CHECK(hipStreamSynchronize(stream));
        }

        // Batch async builds to keep RT cores saturated
        for (int k = 0; k < kernel_loops; k++) {
            OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions, &buildInput, 1,
                                        (CUdeviceptr)d_temp_buffer, bufferSizes.tempSizeInBytes,
                                        (CUdeviceptr)d_output_buffer, bufferSizes.outputSizeInBytes,
                                        &bvh_handle, &emitDesc, 1));
        }
        CHECK(hipStreamSynchronize(stream));
        
        builds += kernel_loops;
        sync_loops++;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << ((builds * num_triangles) / 1e9) / seconds << " GRays/s" << std::endl;

    // --- VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running OptiX BVH Mathematical Bounds Verification..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

        // Verify the 6 properties of the AABB instead of the massive randomized buffer
        LAUNCH_KERNEL_ASYNC(verify_aabb_kernel, 1, 1, 0, stream, d_aabb_buffer, d_golden_aabb, d_err_count);
        CHECK(hipStreamSynchronize(stream));
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        CHECK(hipFree(d_err_count));
        CHECK(hipFree(d_golden_aabb));

        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " errors)" << std::endl;
        if (h_err_count > 0) return 1; 
    }

    OPTIX_CHECK(optixDeviceContextDestroy(context));
    CHECK(hipFree(d_vertices));
    CHECK(hipFree(d_temp_buffer));
    CHECK(hipFree(d_output_buffer));
    CHECK(hipFree(d_aabb_buffer));
    CHECK(hipStreamDestroy(stream));

    return 0;

#elif HIPRT_SUPPORTED
    // ---------------------------------------------------------
    // AMD HIP-RT IMPLEMENTATION
    // ---------------------------------------------------------
    hiprtContextCreationInput ctxInput = {};
    hiprtContext rtContext = nullptr;
    
    if (hiprtCreateContext(HIPRT_API_VERSION, ctxInput, rtContext) != hiprtSuccess) {
        std::cerr << "[PANTHEON ERROR] hiprtCreateContext failed! Ensure libhiprt64.so is installed." << std::endl;
        std::cout << "Throughput: 0.0 GRays/s" << std::endl;
        return 0;
    }

    const int num_triangles = 1000000;
    const int num_vertices = num_triangles * 3;
    size_t num_floats = num_vertices * 3;
    size_t vertices_size = num_floats * sizeof(float);
    size_t indices_size = num_triangles * 3 * sizeof(uint32_t);

    float* d_vertices;
    CHECK(hipMalloc(&d_vertices, vertices_size));
    
    uint32_t* d_indices; 
    CHECK(hipMalloc(&d_indices, indices_size));

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    // Initialize distinct 3D geometry to guarantee BVH determinism
    LAUNCH_KERNEL_ASYNC(init_geometry_kernel, num_blocks, block_size, 0, stream, d_vertices, num_triangles, init_pattern);
    // Initialize required index buffer for AMD HIP-RT
    LAUNCH_KERNEL_ASYNC(init_indices_kernel, num_blocks, block_size, 0, stream, d_indices, num_triangles);
    CHECK(hipStreamSynchronize(stream));

    hiprtTriangleMeshPrimitive mesh = {};
    mesh.vertices = d_vertices;
    mesh.vertexCount = num_vertices;
    mesh.vertexStride = 3 * sizeof(float);
    mesh.triangleCount = num_triangles;

    mesh.triangleIndices = d_indices;
    mesh.triangleStride = 3 * sizeof(uint32_t);

    hiprtGeometryBuildInput buildInput = {};
    buildInput.type = hiprtPrimitiveTypeTriangleMesh;
    buildInput.primitive.triangleMesh = mesh;

    hiprtBuildOptions options = {};
    options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    size_t geomTempSize = 0;
    if (hiprtGetGeometryBuildTemporaryBufferSize(rtContext, buildInput, options, geomTempSize) != hiprtSuccess) {
         std::cerr << "[PANTHEON ERROR] Failed to get HIP-RT memory requirements." << std::endl;
         return 1;
    }

    void* d_temp_buffer;
    CHECK(hipMalloc(&d_temp_buffer, geomTempSize));

    hiprtGeometry geometry = nullptr;
    if (hiprtCreateGeometry(rtContext, buildInput, options, geometry) != hiprtSuccess) {
         std::cerr << "[PANTHEON ERROR] Failed to create HIP-RT geometry." << std::endl;
         return 1;
    }

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running RT VIRUS (HIP-RT BVH Builder)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << " (BVH Builds per Sync)" << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Geometry Scale/Spread)" << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    void* d_golden_temp = nullptr;
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected HIP-RT Builder baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_temp, geomTempSize));
        
        // Zero output buffer to wipe any struct padding that might throw false positives
        CHECK(hipMemsetAsync(d_temp_buffer, 0, geomTempSize, stream));
        
        hiprtBuildGeometry(rtContext, hiprtBuildOperationBuild, buildInput, options, d_temp_buffer, stream, geometry);
        CHECK(hipStreamSynchronize(stream));
        
        CHECK(hipMemcpy(d_golden_temp, d_temp_buffer, geomTempSize, hipMemcpyDeviceToDevice));
    }

    // --- WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup BVH builds..." << std::endl;
        for (int w = 0; w < warmup_iters; w++) {
            CHECK(hipMemsetAsync(d_temp_buffer, 0, geomTempSize, stream));
            hiprtBuildGeometry(rtContext, hiprtBuildOperationBuild, buildInput, options, d_temp_buffer, stream, geometry);
        }
        CHECK(hipStreamSynchronize(stream));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t builds = 0;
    unsigned int sync_loops = 0;

    // --- ACTIVE LOOP ---
    while(true) {
        if (inject_error && sync_loops == 1) {
            LAUNCH_KERNEL_ASYNC(inject_rt_error_kernel, 1, 1, 0, stream, d_vertices, 1337);
            CHECK(hipStreamSynchronize(stream));
        }

        // Batch async builds to keep RT cores saturated
        for (int k = 0; k < kernel_loops; k++) {
            // Wipe struct padding before each build
            CHECK(hipMemsetAsync(d_temp_buffer, 0, geomTempSize, stream));
            hiprtBuildGeometry(rtContext, hiprtBuildOperationBuild, buildInput, options, d_temp_buffer, stream, geometry);
        }
        CHECK(hipStreamSynchronize(stream));
        
        builds += kernel_loops;
        sync_loops++;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << ((builds * num_triangles) / 1e9) / seconds << " GRays/s" << std::endl;

    // --- VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running HIP-RT Builder State Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        size_t num_ints = geomTempSize / sizeof(unsigned int);
        int verify_blocks = (num_ints + 255) / block_size;
        if (verify_blocks > 1024) verify_blocks = 1024;

        LAUNCH_KERNEL(verify_bvh_buffer_kernel, verify_blocks, block_size, (unsigned int*)d_temp_buffer, (unsigned int*)d_golden_temp, num_ints, d_err_count);
        CHECK(hipStreamSynchronize(stream));
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        
        CHECK(hipFree(d_err_count));
        CHECK(hipFree(d_golden_temp));

        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " errors)" << std::endl;
        if (h_err_count > 0) return 1; 
    }

    hiprtDestroyGeometry(rtContext, geometry);
    hiprtDestroyContext(rtContext);
    CHECK(hipFree(d_vertices));
    CHECK(hipFree(d_indices));
    CHECK(hipFree(d_temp_buffer));
    CHECK(hipStreamDestroy(stream));

    return 0;

#else
    // ---------------------------------------------------------
    // FALLBACK
    // ---------------------------------------------------------
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Skipping RT VIRUS (Requires NVIDIA OptiX or AMD HIP-RT headers)." << std::endl;
    if (verify_mode) {
        // State this explicitly: a requested verification that cannot run
        // must never look like one that ran and passed.
        std::cout << "Verification: SKIPPED (no ray tracing backend available)" << std::endl;
    }
    std::cout << "Throughput: 0.0 GRays/s" << std::endl;
    return 0;
#endif
}
