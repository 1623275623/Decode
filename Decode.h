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
	std::string input_dir;
	int batch_size;
	int total_images;
	int dev;
	int warmup;

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

	nvjpegOutputFormat_t fmt;
	bool write_decoded;
	std::string output_dir;

	
};


class Decode
{
public:
	decode_params_t params;


	Decode();


	
	int DecodeImage(unsigned char* data, size_t length, nvjpegImage_t* image,cudaStream_t& stream);


	~Decode();
};

