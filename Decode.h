#pragma once
#include <string>
#include <nvjpeg.h>
#include <iostream>

#pragma comment(lib,"nvjpeg.lib")


int host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }

int host_free(void* p) { return (int)cudaFreeHost(p); }

int dev_malloc(void** p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void* p) { return (int)cudaFree(p); }


nvjpegDevAllocator_t dev_allocator = { &dev_malloc, &dev_free };
nvjpegPinnedAllocator_t pinned_allocator = { &host_malloc, &host_free };

#define CHECK_CUDA(call)                                                                                \
    {                                                                                                       \
        cudaError_t _e = (call);                                                                       \
        if ( _e != cudaSuccess)                                                                  \
        {                                                                                                   \
            std::cout << "CUDA failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
                                                                                        \
        }                                                                                                   \
    }


#define CHECK_NVJPEG(call)                                                                                \
    {                                                                                                       \
        nvjpegStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
                                                                                        \
        }                                                                                                   \
    }


struct decode_params_t {



	nvjpegJpegState_t nvjpeg_state;
	nvjpegHandle_t nvjpeg_handle;
	cudaStream_t stream;

	// used with decoupled API
	nvjpegJpegState_t nvjpeg_decoupled_state;
	nvjpegBufferPinned_t pinned_buffers; // 2 buffers for pipelining
	nvjpegBufferDevice_t device_buffer;
	nvjpegJpegStream_t  jpeg_streams; //  2 streams for pipelining
	nvjpegDecodeParams_t nvjpeg_decode_params;
	nvjpegJpegDecoder_t nvjpeg_decoder;

};


class Decode
{
public:
	decode_params_t params;


	Decode(nvjpegOutputFormat_t output_format=NVJPEG_OUTPUT_BGRI, nvjpegBackend_t Bankend=NVJPEG_BACKEND_GPU_HYBRID);


	int DecodeImage(unsigned char* data, size_t length, nvjpegImage_t* image,nvjpegOutputFormat_t output_fmt,cudaStream_t& stream);


	~Decode();
};

