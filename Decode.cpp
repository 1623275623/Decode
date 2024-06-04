#include "Decode.h"

Decode::Decode()
{
	CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, &dev_allocator, &pinned_allocator, 0, &params.nvjpeg_handle));
	CHECK_NVJPEG(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_GPU_HYBRID, &params.nvjpeg_decoder));
	CHECK_NVJPEG(nvjpegDecoderStateCreate(params.nvjpeg_handle, params.nvjpeg_decoder, &params.nvjpeg_decoupled_state));

	CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL, &params.pinned_buffers));

	CHECK_NVJPEG(nvjpegBufferDeviceCreate(params.nvjpeg_handle, NULL, &params.device_buffer));

	CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams));
	//CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[1]));

	CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &params.nvjpeg_decode_params));
    params.fmt = NVJPEG_OUTPUT_BGRI;
}




int Decode::DecodeImage(unsigned char* data,size_t length,nvjpegImage_t* image,cudaStream_t& stream)
{

    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.fmt));
 
        CHECK_NVJPEG(
           nvjpegJpegStreamParse(params.nvjpeg_handle, (const unsigned char*)data, length,
                0, 0, params.jpeg_streams));

        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
        CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
            params.pinned_buffers));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
            params.nvjpeg_decode_params, params.jpeg_streams));

        //CHECK_NVJPEG(cudaStreamSynchronize(params.stream));

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
            params.jpeg_streams, stream));

        // buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
            image, stream));


	return 0;
}

Decode::~Decode()
{



}





#include <fstream>

int main()
{
    std::ifstream input_file("G:\\525\\NewLevelSequence.0000.jpg", std::ios::in | std::ios::binary | std::ios::ate);
    std::streamsize size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    
    char* data = new char[size];

    input_file.read(data, size);

    Decode decode;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    nvjpegImage_t image;
    decode.DecodeImage((unsigned char*)data, size, &image,stream);



}