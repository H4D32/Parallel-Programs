//
// Created by Zhang Na on 2023/9/15.
// Email: nazhang@link.cuhk.edu.cn
//
// OpenMP implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>    // OpenMP header
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};



int main(int argc, char** argv) {

    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]); // User-specified thread count
    
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);

    // Separate R, G, B channels into three continuous arrays
    auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Transforming the R, G, B channels
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for default(none) shared(rChannel, gChannel, bChannel, filteredImage, input_jpeg, filter) num_threads(num_threads)
    for (int height = 1; height < input_jpeg.height - 1; height++)
    {
        for (int width = 1; width < input_jpeg.width - 1; width++)
        {
            //0 0
            int channel_value_r1 = rChannel[(height-1) * input_jpeg.width + (width-1)];
            int channel_value_g1 = gChannel[(height-1) * input_jpeg.width + (width-1)];
            int channel_value_b1 = bChannel[(height-1) * input_jpeg.width + (width-1)];
            //1 0
            int channel_value_r2 = rChannel[(height) * input_jpeg.width + (width-1)];
            int channel_value_g2 = gChannel[(height) * input_jpeg.width + (width-1)];
            int channel_value_b2 = bChannel[(height) * input_jpeg.width + (width-1)];
            //2 0
            int channel_value_r3 = rChannel[(height+1) * input_jpeg.width + (width-1)];
            int channel_value_g3 = gChannel[(height+1) * input_jpeg.width + (width-1)];
            int channel_value_b3 = bChannel[(height+1) * input_jpeg.width + (width-1)];
            //0 1
            int channel_value_r4 = rChannel[(height-1) * input_jpeg.width + (width)];
            int channel_value_g4 = gChannel[(height-1) * input_jpeg.width + (width)];
            int channel_value_b4 = bChannel[(height-1) * input_jpeg.width + (width)];
            //1 1
            int channel_value_r5 = rChannel[(height) * input_jpeg.width + (width)];
            int channel_value_g5 = gChannel[(height) * input_jpeg.width + (width)];
            int channel_value_b5 = bChannel[(height) * input_jpeg.width + (width)];
            //2 1
            int channel_value_r6 = rChannel[(height+1) * input_jpeg.width + (width)];
            int channel_value_g6 = gChannel[(height+1) * input_jpeg.width + (width)];
            int channel_value_b6 = bChannel[(height+1) * input_jpeg.width + (width)];
            //0 2
            int channel_value_r7 = rChannel[(height-1) * input_jpeg.width + (width+1)];
            int channel_value_g7 = gChannel[(height-1) * input_jpeg.width + (width+1)];
            int channel_value_b7 = bChannel[(height-1) * input_jpeg.width + (width+1)];
            //1 2
            int channel_value_r8 = rChannel[(height) * input_jpeg.width + (width+1)];
            int channel_value_g8 = gChannel[(height) * input_jpeg.width + (width+1)];
            int channel_value_b8 = bChannel[(height) * input_jpeg.width + (width+1)];
            //2 2
            int channel_value_r9 = rChannel[(height+1) * input_jpeg.width + (width+1)];
            int channel_value_g9 = gChannel[(height+1) * input_jpeg.width + (width+1)];
            int channel_value_b9 = bChannel[(height+1) * input_jpeg.width + (width+1)];

            int sum_r = channel_value_r1 * filter[0][0]+ channel_value_r2 * filter[1][0]+ channel_value_r3 * filter[2][0]+
                        channel_value_r4 * filter[0][1]+ channel_value_r5 * filter[1][1]+ channel_value_r6 * filter[2][1]+
                        channel_value_r7 * filter[0][2]+ channel_value_r8 * filter[1][2]+ channel_value_r9 * filter[2][2];
            
            int sum_g = channel_value_g1 * filter[0][0]+ channel_value_g2 * filter[1][0]+ channel_value_g3 * filter[2][0]+
                        channel_value_g4 * filter[0][1]+ channel_value_g5 * filter[1][1]+ channel_value_g6 * filter[2][1]+
                        channel_value_g7 * filter[0][2]+ channel_value_g8 * filter[1][2]+ channel_value_g9 * filter[2][2];

            int sum_b = channel_value_b1 * filter[0][0]+ channel_value_b2 * filter[1][0]+ channel_value_b3 * filter[2][0]+
                        channel_value_b4 * filter[0][1]+ channel_value_b5 * filter[1][1]+ channel_value_b6 * filter[2][1]+
                        channel_value_b7 * filter[0][2]+ channel_value_b8 * filter[1][2]+ channel_value_b9 * filter[2][2];

            filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels]
                = static_cast<unsigned char>(std::round(sum_r));
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
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release the allocated memory
    delete[] input_jpeg.buffer;
    delete[] rChannel;
    delete[] gChannel;
    delete[] bChannel;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
