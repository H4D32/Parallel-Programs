//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <iostream>
#include <cmath>
#include <chrono>

#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    // Nested for loop, please optimize it
    for (int height = 1; height < input_jpeg.height - 1; height++)
    {
        for (int width = 1; width < input_jpeg.width - 1; width++)
        {
            //Unrolled 9-length loop
            //0 0
            int channel_value_r1 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width-1)) * input_jpeg.num_channels] ;
            int channel_value_g1 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width-1)) * input_jpeg.num_channels + 1];
            int channel_value_b1 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width-1)) * input_jpeg.num_channels + 2];
            //1 0
            int channel_value_r2 = input_jpeg.buffer[((height) * input_jpeg.width + (width-1)) * input_jpeg.num_channels];
            int channel_value_g2 = input_jpeg.buffer[((height) * input_jpeg.width + (width-1)) * input_jpeg.num_channels + 1];
            int channel_value_b2 = input_jpeg.buffer[((height) * input_jpeg.width + (width-1)) * input_jpeg.num_channels + 2];
            //2 0
            int channel_value_r3 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width-1)) * input_jpeg.num_channels];
            int channel_value_g3 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width-1)) * input_jpeg.num_channels + 1];
            int channel_value_b3 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width-1)) * input_jpeg.num_channels + 2];
            //0 1
            int channel_value_r4 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width)) * input_jpeg.num_channels];
            int channel_value_g4 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width)) * input_jpeg.num_channels + 1];
            int channel_value_b4 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width)) * input_jpeg.num_channels + 2];
            //1 1
            int channel_value_r5 = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels];
            int channel_value_g5 = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels + 1];
            int channel_value_b5 = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels + 2];
            //2 1
            int channel_value_r6 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width)) * input_jpeg.num_channels];
            int channel_value_g6 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width)) * input_jpeg.num_channels + 1];
            int channel_value_b6 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width)) * input_jpeg.num_channels + 2];
            //0 2
            int channel_value_r7 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width+1)) * input_jpeg.num_channels];
            int channel_value_g7 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width+1)) * input_jpeg.num_channels + 1];
            int channel_value_b7 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width+1)) * input_jpeg.num_channels + 2];
            //1 2
            int channel_value_r8 = input_jpeg.buffer[((height) * input_jpeg.width + (width+1)) * input_jpeg.num_channels];
            int channel_value_g8 = input_jpeg.buffer[((height) * input_jpeg.width + (width+1)) * input_jpeg.num_channels + 1];
            int channel_value_b8 = input_jpeg.buffer[((height) * input_jpeg.width + (width+1)) * input_jpeg.num_channels + 2];
            //2 2
            int channel_value_r9 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width+1)) * input_jpeg.num_channels];
            int channel_value_g9 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width+1)) * input_jpeg.num_channels + 1];
            int channel_value_b9 = input_jpeg.buffer[((height+1) * input_jpeg.width + (width+1)) * input_jpeg.num_channels + 2];

            int sum_r = channel_value_r1 * filter[0][0]+ channel_value_r2 * filter[1][0]+ channel_value_r3 * filter[2][0]+
                        channel_value_r4 * filter[0][1]+ channel_value_r5 * filter[1][1]+ channel_value_r6 * filter[2][1]+
                        channel_value_r7 * filter[0][2]+ channel_value_r8 * filter[1][2]+ channel_value_r9 * filter[2][2];
            
            int sum_g = channel_value_g1 * filter[0][0]+ channel_value_g2 * filter[1][0]+ channel_value_g3 * filter[2][0]+
                        channel_value_g4 * filter[0][1]+ channel_value_g5 * filter[1][1]+ channel_value_g6 * filter[2][1]+
                        channel_value_g7 * filter[0][2]+ channel_value_g8 * filter[1][2]+ channel_value_g9 * filter[2][2];

            int sum_b = channel_value_b1 * filter[0][0]+ channel_value_b2 * filter[1][0]+ channel_value_b3 * filter[2][0]+
                        channel_value_b4 * filter[0][1]+ channel_value_b5 * filter[1][1]+ channel_value_b6 * filter[2][1]+
                        channel_value_b7 * filter[0][2]+ channel_value_b8 * filter[1][2]+ channel_value_b9 * filter[2][2];
                        
                            // Back to base
                            filteredImage[(height * input_jpeg.width + width) *
                                          input_jpeg.num_channels] =
                            static_cast<unsigned char>(std::round(sum_r));
            filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
