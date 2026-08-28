#include "../common/common.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#if defined(__CUDACC__)
    #include <cuda.h>
    #include <dlfcn.h>
    // Suppress the harmless bitfield warnings from the FFmpeg header
    #pragma diag_suppress 108
    #include "nvEncodeAPI.h"
    #define NVENC_SUPPORTED 1
#else
    #define NVENC_SUPPORTED 0
#endif

#if NVENC_SUPPORTED
#define NVENC_CHECK(call) \
    { \
        NVENCSTATUS res = call; \
        if (res != NV_ENC_SUCCESS && res != NV_ENC_ERR_NEED_MORE_INPUT) { \
            std::cerr << "[PANTHEON ERROR] NVENC call failed with code " << res << " at line " << __LINE__ << std::endl; \
            exit(2); \
        } \
    }

typedef NVENCSTATUS(NVENCAPI *PNVENCODEAPICREATEINSTANCE)(NV_ENCODE_API_FUNCTION_LIST*);

// --- CUDA KERNELS FOR INPUT BUFFER MANAGEMENT ---
__global__ void init_frame_kernel(unsigned char* ptr, int width, int height, int pitch) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t total_pixels = width * height;
    
    for (size_t i = idx; i < total_pixels; i += stride) {
        int x = i % width;
        int y = i / width;
        // Generate a deterministic ARGB gradient pattern
        uint32_t* pixel = (uint32_t*)(ptr + y * pitch + x * 4);
        *pixel = 0xFF000000 | ((x & 0xFF) << 16) | ((y & 0xFF) << 8) | ((x ^ y) & 0xFF);
    }
}

__global__ void inject_frame_error_kernel(unsigned char* ptr, int pitch) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Intentionally corrupt a specific pixel at coordinates (1337, 1337)
        uint32_t* pixel = (uint32_t*)(ptr + 1337 * pitch + 1337 * 4);
        *pixel ^= 0x00FFFFFF;
    }
}
#endif

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    bool verify_mode = false;
    int inject_error = 0;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verify") verify_mode = true;
        if (std::string(argv[i]) == "--inject_error") inject_error = 1;
    }

    CHECK(hipSetDevice(gpu_id));

#if !NVENC_SUPPORTED
    std::cerr << "[PANTHEON] GPU " << gpu_id << ": Skipping MEDIA_ENC_VIRUS (Requires NVIDIA CUDA)." << std::endl;
    std::cout << "Throughput: 0.0 FPS" << std::endl;
    return 0;
#else

    // 1. Initialize CUDA Driver and Hook Primary Context
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "[PANTHEON ERROR] cuInit(0) failed." << std::endl;
        std::cout << "Throughput: 0.0 FPS" << std::endl;
        return 0;
    }

    CUdevice cuDev = 0;
    cuDeviceGet(&cuDev, gpu_id);
    CUcontext cuCtx = nullptr;
    // Bypasses the cuCtxCreate_v4 macro bug by retaining the context hipSetDevice just made
    cuDevicePrimaryCtxRetain(&cuCtx, cuDev);

    // 2. Dynamically Load the NVIDIA Encode Driver
    void* encode_lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (!encode_lib) {
        std::cerr << "[PANTHEON ERROR] libnvidia-encode.so.1 not found! Is the driver installed?" << std::endl;
        std::cout << "Throughput: 0.0 FPS" << std::endl;
        return 0;
    }

    auto NvEncCreate = (PNVENCODEAPICREATEINSTANCE)dlsym(encode_lib, "NvEncodeAPICreateInstance");
    
    NV_ENCODE_API_FUNCTION_LIST nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };
    NVENC_CHECK(NvEncCreate(&nvenc));

    // 3. Open Encode Session
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
    sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    sessionParams.device = cuCtx;
    sessionParams.apiVersion = NVENCAPI_VERSION;
    void* encoder = nullptr;
    NVENC_CHECK(nvenc.nvEncOpenEncodeSessionEx(&sessionParams, &encoder));

    // 4. Initialize 4K HEVC Constant Quality Preset (P7)
    NV_ENC_INITIALIZE_PARAMS initParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    initParams.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
    initParams.presetGUID = NV_ENC_PRESET_P7_GUID;
    initParams.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    initParams.encodeWidth = 3840;
    initParams.encodeHeight = 2160;
    initParams.maxEncodeWidth = 3840;
    initParams.maxEncodeHeight = 2160;
    initParams.darWidth = 3840;
    initParams.darHeight = 2160;
    initParams.frameRateNum = 60;
    initParams.frameRateDen = 1;
    initParams.enablePTD = 1;

    NV_ENC_PRESET_CONFIG presetConfig = { 0 };
    presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
    presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;

    NVENC_CHECK(nvenc.nvEncGetEncodePresetConfigEx(encoder, initParams.encodeGUID, initParams.presetGUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &presetConfig));
    
    // OVERRIDE: Force Constant QP to guarantee perfectly deterministic bitstream sizes
    presetConfig.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    presetConfig.presetCfg.rcParams.constQP.qpInterP = 28;
    presetConfig.presetCfg.rcParams.constQP.qpInterB = 28;
    presetConfig.presetCfg.rcParams.constQP.qpIntra = 28;
    
    initParams.encodeConfig = &presetConfig.presetCfg;
    NVENC_CHECK(nvenc.nvEncInitializeEncoder(encoder, &initParams));

    // 5. Allocate Dummy Input and Output Buffers
    NV_ENC_CREATE_INPUT_BUFFER inputBuf = { NV_ENC_CREATE_INPUT_BUFFER_VER };
    inputBuf.width = 3840;
    inputBuf.height = 2160;
    inputBuf.bufferFmt = NV_ENC_BUFFER_FORMAT_ARGB;
    NVENC_CHECK(nvenc.nvEncCreateInputBuffer(encoder, &inputBuf));

    NV_ENC_CREATE_BITSTREAM_BUFFER bitBuf = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
    NVENC_CHECK(nvenc.nvEncCreateBitstreamBuffer(encoder, &bitBuf));

    // 6. Initialize the NVENC Memory Buffer via CUDA
    NV_ENC_LOCK_INPUT_BUFFER lockInputInit = { NV_ENC_LOCK_INPUT_BUFFER_VER };
    lockInputInit.inputBuffer = inputBuf.inputBuffer;
    NVENC_CHECK(nvenc.nvEncLockInputBuffer(encoder, &lockInputInit));

    hipDeviceProp_t prop; 
    CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 16;
    
    LAUNCH_KERNEL(init_frame_kernel, num_blocks, 256, (unsigned char*)lockInputInit.bufferDataPtr, 3840, 2160, lockInputInit.pitch);
    CHECK(hipDeviceSynchronize());
    NVENC_CHECK(nvenc.nvEncUnlockInputBuffer(encoder, inputBuf.inputBuffer));

    std::cerr << "[PANTHEON] GPU " << gpu_id << ": NVENC Session Bound. Spiking HEVC Encoder..." << std::endl;
    if (inject_error) std::cerr << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
    picParams.inputWidth = 3840;
    picParams.inputHeight = 2160;
    picParams.inputPitch = lockInputInit.pitch; 
    picParams.inputBuffer = inputBuf.inputBuffer;
    picParams.outputBitstream = bitBuf.bitstreamBuffer;
    picParams.bufferFmt = NV_ENC_BUFFER_FORMAT_ARGB;
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    // FORCE IDR: Makes every frame 100% independent and identically compressed
    picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;

    // --- GOLDEN PASS ---
    std::vector<uint8_t> golden_bitstream;
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected NVENC Bitstream baseline (Golden Pass)..." << std::endl;
        NVENC_CHECK(nvenc.nvEncEncodePicture(encoder, &picParams));
        
        NV_ENC_LOCK_BITSTREAM lockBs = { NV_ENC_LOCK_BITSTREAM_VER };
        lockBs.outputBitstream = bitBuf.bitstreamBuffer;
        NVENC_CHECK(nvenc.nvEncLockBitstream(encoder, &lockBs));
        
        golden_bitstream.assign((uint8_t*)lockBs.bitstreamBufferPtr, (uint8_t*)lockBs.bitstreamBufferPtr + lockBs.bitstreamSizeInBytes);
        NVENC_CHECK(nvenc.nvEncUnlockBitstream(encoder, bitBuf.bitstreamBuffer));
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t frames_encoded = 0;
    unsigned int err_count = 0;
    // Bitstream comparison is host-side work, not encoder throughput; track
    // its cost so verified runs report the same FPS as unverified ones.
    double verify_overhead_s = 0.0;

    // 7. The Execution Loop
    while(true) {
        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && frames_encoded == 20) {
            NV_ENC_LOCK_INPUT_BUFFER lockInputInj = { NV_ENC_LOCK_INPUT_BUFFER_VER };
            lockInputInj.inputBuffer = inputBuf.inputBuffer;
            NVENC_CHECK(nvenc.nvEncLockInputBuffer(encoder, &lockInputInj));
            
            LAUNCH_KERNEL(inject_frame_error_kernel, 1, 1, (unsigned char*)lockInputInj.bufferDataPtr, lockInputInj.pitch);
            CHECK(hipDeviceSynchronize());
            NVENC_CHECK(nvenc.nvEncUnlockInputBuffer(encoder, inputBuf.inputBuffer));
        }

        // Force the hardware to attempt compressing VRAM
        NVENC_CHECK(nvenc.nvEncEncodePicture(encoder, &picParams));

        // --- VERIFICATION CHECK ---
        if (verify_mode) {
            NV_ENC_LOCK_BITSTREAM lockBs = { NV_ENC_LOCK_BITSTREAM_VER };
            lockBs.outputBitstream = bitBuf.bitstreamBuffer;
            // Locking blocks until the encoder has finished the frame. That
            // wait is encode time, not verification, so it must stay inside
            // the FPS clock -- subtracting it would report the rate at which
            // frames are submitted rather than encoded.
            NVENC_CHECK(nvenc.nvEncLockBitstream(encoder, &lockBs));

            // Time only the host-side bitstream comparison.
            auto verify_start = std::chrono::high_resolution_clock::now();
            if (lockBs.bitstreamSizeInBytes != golden_bitstream.size()) {
                if (err_count < 5) printf("[SDC FAULT][MEDIA_ENC] Bitstream Size Mismatch! Exp: %zu | Act: %u\n", golden_bitstream.size(), lockBs.bitstreamSizeInBytes);
                err_count++;
            } else {
                if (memcmp(lockBs.bitstreamBufferPtr, golden_bitstream.data(), golden_bitstream.size()) != 0) {
                    if (err_count < 5) printf("[SDC FAULT][MEDIA_ENC] Bitstream Payload Corruption detected!\n");
                    err_count++;
                }
            }
            verify_overhead_s += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - verify_start).count();

            NVENC_CHECK(nvenc.nvEncUnlockBitstream(encoder, bitBuf.bitstreamBuffer));
        }

        frames_encoded++;
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    seconds -= verify_overhead_s;
    if (seconds <= 0.0) seconds = 1e-6;
    std::cout << "Throughput: " << (frames_encoded / seconds) << " FPS" << std::endl;

    // Cleanup
    nvenc.nvEncDestroyInputBuffer(encoder, inputBuf.inputBuffer);
    nvenc.nvEncDestroyBitstreamBuffer(encoder, bitBuf.bitstreamBuffer);
    nvenc.nvEncDestroyEncoder(encoder);
    dlclose(encode_lib);
    cuDevicePrimaryCtxRelease(cuDev); 

    // CRITICAL: Exit with non-zero code to fail the CI step if errors were found
    // Say so on success too. Silence is indistinguishable from a
    // verification that never ran.
    std::cout << "Verification: " << (err_count ? "FAIL" : "PASS")
              << " (" << err_count << " errors)" << std::endl;
    if (err_count > 0) return 1;

    return 0;
#endif
}
