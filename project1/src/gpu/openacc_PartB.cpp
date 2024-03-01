//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};
const float F00 = filter[0][0];
const float F01 = filter[0][1];
const float F02 = filter[0][2];
const float F10 = filter[1][0];
const float F11 = filter[1][1];
const float F12 = filter[1][2];
const float F20 = filter[2][0];
const float F21 = filter[2][1];
const float F22 = filter[2][2];


int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
    // Computation: RGB to Gray
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *filteredImage = new unsigned char[width * height * num_channels];
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
#pragma acc enter data copyin(filteredImage[0 : width * height * num_channels], \
                              buffer[0 : width * height * num_channels])

#pragma acc update device(filteredImage[0 : width * height * num_channels], \
                          buffer[0 : width * height * num_channels])

    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel present(filteredImage[0 : width * height * num_channels], \
                             buffer[0 : width * height * num_channels])        \
    async(1)
    {
#pragma acc loop independent
//Loop Code
        for (int X = width + 1; X < height * width; X++)
        {   
                int a00 = X - width - 1;             // X - width - 1
                int a10 = X - 1;                     // X - 1
                int a20 = X + width - 1;             // X + width - 1
                int a01 = X - width;                 // X - width
                int a11 = X;                         // X
                int a21 = X + width;                 // X + width
                int a02 = X - width + 1;             // X - width + 1
                int a12 = X + 1;                     // X + 1
                int a22 = X + width + 1;             // X + width + 1
                // Back to base
                filteredImage[X * num_channels] =
                                   buffer[(a00) * num_channels] * F00 +
                                   buffer[(a10) * num_channels] * F10 +
                                   buffer[(a20) * num_channels] * F20 +
                                   buffer[(a01) * num_channels] * F01 +
                                   buffer[(a11) * num_channels] * F11 +
                                   buffer[(a21) * num_channels] * F21 +
                                   buffer[(a02) * num_channels] * F02 +
                                   buffer[(a12) * num_channels] * F12 +
                                   buffer[(a22) * num_channels] * F22;
                filteredImage[X * num_channels + 1] =
                                   buffer[(a00) * num_channels + 1] * F00 +
                                   buffer[(a10) * num_channels + 1] * F10 +
                                   buffer[(a20) * num_channels + 1] * F20 +
                                   buffer[(a01) * num_channels + 1] * F01 +
                                   buffer[(a11) * num_channels + 1] * F11 +
                                   buffer[(a21) * num_channels + 1] * F21 +
                                   buffer[(a02) * num_channels + 1] * F02 +
                                   buffer[(a12) * num_channels + 1] * F12 +
                                   buffer[(a22) * num_channels + 1] * F22;
                filteredImage[X * num_channels + 2] =
                                   buffer[(a00) * num_channels + 2] * F00 +
                                   buffer[(a10) * num_channels + 2] * F10 +
                                   buffer[(a20) * num_channels + 2] * F20 +
                                   buffer[(a01) * num_channels + 2] * F01 +
                                   buffer[(a11) * num_channels + 2] * F11 +
                                   buffer[(a21) * num_channels + 2] * F21 +
                                   buffer[(a02) * num_channels + 2] * F02 +
                                   buffer[(a12) * num_channels + 2] * F12 +
                                   buffer[(a22) * num_channels + 2] * F22;
            }             
        }
    auto end_time = std::chrono::high_resolution_clock::now();
#pragma acc update self(filteredImage[0 : width * height * num_channels], \
                        buffer[0 : width * height * num_channels])

#pragma acc exit data copyout(filteredImage[0 : width * height * num_channels])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Write GrayImage to output JPEG
    const char *output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
