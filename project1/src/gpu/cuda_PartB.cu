//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// CUDA implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>

#include <cuda_runtime.h> // CUDA Header

#include "utils.hpp"

// CUDA kernel functon：RGB to Gray
__global__ void rgbSmooth(const unsigned char* input, unsigned char* output,
                          int width, int height, int num_channels,
                          const float* filter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + width + 1;
    if (idx < width * height)
    {
        int a00 = (idx - width - 1) * num_channels; // idx - width - 1 -> Top Left
        int a10 = (idx - 1) * num_channels;         // idx - 1
        int a20 = (idx + width - 1) * num_channels; // idx + width - 1
        int a01 = (idx - width) * num_channels;     // idx - width
        int a11 = idx * num_channels;             // idx
        int a21 = (idx + width) * num_channels;     // idx + width
        int a02 = (idx - width + 1) * num_channels; // idx - width + 1
        int a12 = (idx + 1) * num_channels;         // idx + 1
        int a22 = (idx + width + 1) * num_channels; // idx + width + 1

        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

        sum_r = static_cast<float>(input[a00]) * filter[0] +
                static_cast<float>(input[a01]) * filter[1] +
                static_cast<float>(input[a02]) * filter[2] +
                static_cast<float>(input[a10]) * filter[3] +
                static_cast<float>(input[a11]) * filter[4] +
                static_cast<float>(input[a12]) * filter[5] +
                static_cast<float>(input[a20]) * filter[6] +
                static_cast<float>(input[a21]) * filter[7] +
                static_cast<float>(input[a22]) * filter[8];

        sum_g = static_cast<float>(input[a00 + 1]) * filter[0] +
               static_cast<float>(input[a01 + 1]) * filter[1] +
               static_cast<float>(input[a02 + 1]) * filter[2] +
               static_cast<float>(input[a10 + 1]) * filter[3] +
               static_cast<float>(input[a11 + 1]) * filter[4] +
               static_cast<float>(input[a12 + 1]) * filter[5] +
               static_cast<float>(input[a20 + 1]) * filter[6] +
               static_cast<float>(input[a21 + 1]) * filter[7] +
               static_cast<float>(input[a22 + 1]) * filter[8];

        sum_b = static_cast<float>(input[a00 + 2]) * filter[0] +
               static_cast<float>(input[a01 + 2]) * filter[1] +
               static_cast<float>(input[a02 + 2]) * filter[2] +
               static_cast<float>(input[a10 + 2]) * filter[3] +
               static_cast<float>(input[a11 + 2]) * filter[4] +
               static_cast<float>(input[a12 + 2]) * filter[5] +
               static_cast<float>(input[a20 + 2]) * filter[6] +
               static_cast<float>(input[a21 + 2]) * filter[7] +
               static_cast<float>(input[a22 + 2]) * filter[8];

        output[(idx)*num_channels] = static_cast<unsigned char>(sum_r);
        output[(idx)*num_channels + 1] = static_cast<unsigned char>(sum_g);
        output[(idx)*num_channels + 2] = static_cast<unsigned char>(sum_b);
    }
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels]; 
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_filter;
    cudaMalloc((void**)&d_input, input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels *
                                     sizeof(unsigned char));
    cudaMalloc((void**)&d_output, input_jpeg.width * input_jpeg.height *
                                      input_jpeg.num_channels *
                                      sizeof(unsigned char));
    cudaMalloc((void**)&d_filter, 9 * sizeof(float));
    float array1DFilter[9] = {1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9,
                         1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9};
    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
                   sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, array1DFilter, 9 * sizeof(float),
               cudaMemcpyHostToDevice);
    // Computation: RGB to Gray
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blockSize = 512; // 256
    // int numBlocks =
    //     (input_jpeg.width * input_jpeg.height + blockSize - 1) / blockSize;
    int numBlocks = (input_jpeg.width * input_jpeg.height) / blockSize + 1;
    cudaEventRecord(start, 0); // GPU start time
    rgbSmooth<<<numBlocks, blockSize>>>(d_input, d_output, input_jpeg.width,
                                        input_jpeg.height,
                                        input_jpeg.num_channels, d_filter);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}