#include "Decode.h"

Decode::Decode(nvjpegOutputFormat_t output_format,nvjpegBackend_t Bankend)
{
	CHECK_NVJPEG(nvjpegCreateEx(Bankend, &dev_allocator, &pinned_allocator, 0, &params.nvjpeg_handle));
	CHECK_NVJPEG(nvjpegDecoderCreate(params.nvjpeg_handle, Bankend, &params.nvjpeg_decoder));
	CHECK_NVJPEG(nvjpegDecoderStateCreate(params.nvjpeg_handle, params.nvjpeg_decoder, &params.nvjpeg_decoupled_state));

	CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL, &params.pinned_buffers));

	CHECK_NVJPEG(nvjpegBufferDeviceCreate(params.nvjpeg_handle, NULL, &params.device_buffer));

	CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams));
	//CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[1]));
	CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &params.nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params,output_format));
}




int Decode::DecodeImage(unsigned char* data, size_t length, nvjpegImage_t * image, nvjpegOutputFormat_t output_fmt, cudaStream_t & stream)
{

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    cudaEventRecord(start,stream);
        CHECK_NVJPEG(
           nvjpegJpegStreamParse(params.nvjpeg_handle, (const unsigned char*)data, length,
                0, 0, params.jpeg_streams));

        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
        CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
            params.pinned_buffers));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
            params.nvjpeg_decode_params, params.jpeg_streams));

        cudaStreamSynchronize(stream);

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
            params.jpeg_streams, stream));

        // buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
            image, stream));

        CHECK_CUDA(cudaEventRecord(stop,stream));
        float time = 0;
        cudaStreamSynchronize(stream);
        CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
        std::cout << time;
	return 0;
}

Decode::~Decode()
{
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(params.nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers));
    CHECK_NVJPEG(nvjpegBufferDeviceDestroy(params.device_buffer));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_decoupled_state));
    CHECK_NVJPEG(nvjpegDecoderDestroy(params.nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));
   
}





#include <fstream>
#include <opencv2/opencv.hpp>
#include <memory>
int main()
{
    
    std::ifstream input_file("test.jpg", std::ios::in | std::ios::binary | std::ios::ate); //��ͼƬ�Ӵ��̵��ж�ȡ���� 
    std::streamsize size = input_file.tellg();  //��ȡ������ȡ�ļ��Ĵ�С
    input_file.seekg(0, std::ios::beg);
    
    char* data = new char[size]; //Ϊ�洢ͼƬ���ݴ����洢�ռ�
    

    //ʹ��
    input_file.read(data, size);  //��ȡͼƬ���ݣ��������洢�ռ�


    //cv::imshow("display", s);
    Decode decode;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); //������
    nvjpegImage_t image;
    cudaMalloc((void**)&image.channel[0], 3840 * 2160 * 3);   //�����ռ�������Ž�����ͼ������
    image.pitch[0] = 3840 * 3;
    decode.DecodeImage((unsigned char*)data, size, &image,NVJPEG_OUTPUT_BGRI,stream); //���ú�������ͼ��Ľ��� ǰ���������ֱ���ͼ�����ݣ��Լ�ͼ�����ݵĳ���



    unsigned char* data2 = new unsigned char[3840 * 2160 * 3];  //����һ���µ����� 
    /*checkCudaErrors(cudaMemcpy2D(chanRGB, (size_t)width * 3, d_RGB, (size_t)pitch,
        width * 3, height, cudaMemcpyDeviceToHost));*/
    
     //�����ݴ�GPU������CPU
    cudaMemcpy2D(data2,size_t(3840*3),image.channel[0],image.pitch[0],3840*3,2160, cudaMemcpyDeviceToHost);
   // cv::imshow("asd", img);
    //std::destroy_at(&decode); 
    
    //�����������д�����̵���
    cv::Mat img(2160,3840,CV_8UC3, data2);
    cv::imwrite("out.jpg", img);
   // int i = 0;
    //std::cin >> i;


    //�Դ��������ݽ����ͷ�
    delete[] data;
    delete[] data2;
    cudaFree(image.channel[0]);

}