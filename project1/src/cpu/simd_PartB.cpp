//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <immintrin.h>


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

    auto filteredImage =
        new unsigned char[input_jpeg.width * input_jpeg.height *
                          input_jpeg.num_channels];
    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    
    auto redSmooth = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greenSmooth = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blueSmooth = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Set SIMD scalars, we use AVX2 instructions

    __m256 Scalar00 = _mm256_set1_ps(static_cast<float>(filter[0][0]));
    __m256 Scalar01 = _mm256_set1_ps(static_cast<float>(filter[0][1]));
    __m256 Scalar02 = _mm256_set1_ps(static_cast<float>(filter[0][2]));
    __m256 Scalar10 = _mm256_set1_ps(static_cast<float>(filter[1][0]));
    __m256 Scalar11 = _mm256_set1_ps(static_cast<float>(filter[1][1]));
    __m256 Scalar12 = _mm256_set1_ps(static_cast<float>(filter[1][2]));
    __m256 Scalar20 = _mm256_set1_ps(static_cast<float>(filter[2][0]));
    __m256 Scalar21 = _mm256_set1_ps(static_cast<float>(filter[2][1]));
    __m256 Scalar22 = _mm256_set1_ps(static_cast<float>(filter[2][2]));


    // Mask used for shuffling when store int32s to u_int8 arrays
    // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
    __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int X = input_jpeg.width + 1; X < input_jpeg.width * input_jpeg.height; X += 8)
    {
        __m128i red00_chars = _mm_loadu_si128((__m128i*) (reds+X - input_jpeg.width-1));
        __m256i red00_ints = _mm256_cvtepu8_epi32(red00_chars);
        __m256 red00_floats = _mm256_cvtepi32_ps(red00_ints);
        __m256 red00_results = _mm256_mul_ps(red00_floats, Scalar00);

        __m128i green00_chars = _mm_loadu_si128((__m128i*) (greens+X - input_jpeg.width-1));
        __m256i green00_ints = _mm256_cvtepu8_epi32(green00_chars);
        __m256 green00_floats = _mm256_cvtepi32_ps(green00_ints);
        __m256 green00_results = _mm256_mul_ps(green00_floats, Scalar00);

        __m128i blue00_chars = _mm_loadu_si128((__m128i*) (blues+X - input_jpeg.width-1));
        __m256i blue00_ints = _mm256_cvtepu8_epi32(blue00_chars);
        __m256 blue00_floats = _mm256_cvtepi32_ps(blue00_ints);
        __m256 blue00_results = _mm256_mul_ps(blue00_floats, Scalar00);

        __m128i red01_chars = _mm_loadu_si128((__m128i*) (reds+X - input_jpeg.width));
        __m256i red01_ints = _mm256_cvtepu8_epi32(red01_chars);
        __m256 red01_floats = _mm256_cvtepi32_ps(red01_ints);
        __m256 red01_results = _mm256_mul_ps(red01_floats, Scalar01);

        __m128i green01_chars = _mm_loadu_si128((__m128i*) (greens+X - input_jpeg.width));
        __m256i green01_ints = _mm256_cvtepu8_epi32(green01_chars);
        __m256 green01_floats = _mm256_cvtepi32_ps(green01_ints);
        __m256 green01_results = _mm256_mul_ps(green01_floats, Scalar01);

        __m128i blue01_chars = _mm_loadu_si128((__m128i*) (blues+X - input_jpeg.width));
        __m256i blue01_ints = _mm256_cvtepu8_epi32(blue01_chars);
        __m256 blue01_floats = _mm256_cvtepi32_ps(blue01_ints);
        __m256 blue01_results = _mm256_mul_ps(blue01_floats, Scalar01);

        __m128i red02_chars = _mm_loadu_si128((__m128i*) (reds+X - input_jpeg.width+1));
        __m256i red02_ints = _mm256_cvtepu8_epi32(red02_chars);
        __m256 red02_floats = _mm256_cvtepi32_ps(red02_ints);
        __m256 red02_results = _mm256_mul_ps(red02_floats, Scalar02);

        __m128i green02_chars = _mm_loadu_si128((__m128i*) (greens+X - input_jpeg.width+1));
        __m256i green02_ints = _mm256_cvtepu8_epi32(green02_chars);
        __m256 green02_floats = _mm256_cvtepi32_ps(green02_ints);
        __m256 green02_results = _mm256_mul_ps(green02_floats, Scalar02);

        __m128i blue02_chars = _mm_loadu_si128((__m128i*) (blues+X - input_jpeg.width+1));
        __m256i blue02_ints = _mm256_cvtepu8_epi32(blue02_chars);
        __m256 blue02_floats = _mm256_cvtepi32_ps(blue02_ints);
        __m256 blue02_results = _mm256_mul_ps(blue02_floats, Scalar02);


        __m128i red10_chars = _mm_loadu_si128((__m128i*) (reds+X -1));
        __m256i red10_ints = _mm256_cvtepu8_epi32(red10_chars);
        __m256 red10_floats = _mm256_cvtepi32_ps(red10_ints);
        __m256 red10_results = _mm256_mul_ps(red10_floats, Scalar10);

        __m128i green10_chars = _mm_loadu_si128((__m128i*) (greens+X -1));
        __m256i green10_ints = _mm256_cvtepu8_epi32(green10_chars);
        __m256 green10_floats = _mm256_cvtepi32_ps(green10_ints);
        __m256 green10_results = _mm256_mul_ps(green10_floats, Scalar10);

        __m128i blue10_chars = _mm_loadu_si128((__m128i*) (blues+X -1));
        __m256i blue10_ints = _mm256_cvtepu8_epi32(blue10_chars);
        __m256 blue10_floats = _mm256_cvtepi32_ps(blue10_ints);
        __m256 blue10_results = _mm256_mul_ps(blue10_floats, Scalar10);

        __m128i red11_chars = _mm_loadu_si128((__m128i*) (reds+X));
        __m256i red11_ints = _mm256_cvtepu8_epi32(red11_chars);
        __m256 red11_floats = _mm256_cvtepi32_ps(red11_ints);
        __m256 red11_results = _mm256_mul_ps(red11_floats, Scalar11);

        __m128i green11_chars = _mm_loadu_si128((__m128i*) (greens+X));
        __m256i green11_ints = _mm256_cvtepu8_epi32(green11_chars);
        __m256 green11_floats = _mm256_cvtepi32_ps(green11_ints);
        __m256 green11_results = _mm256_mul_ps(green11_floats, Scalar11);

        __m128i blue11_chars = _mm_loadu_si128((__m128i*) (blues+X));
        __m256i blue11_ints = _mm256_cvtepu8_epi32(blue11_chars);
        __m256 blue11_floats = _mm256_cvtepi32_ps(blue11_ints);
        __m256 blue11_results = _mm256_mul_ps(blue11_floats, Scalar11);

        __m128i red12_chars = _mm_loadu_si128((__m128i*) (reds+X +1));
        __m256i red12_ints = _mm256_cvtepu8_epi32(red12_chars);
        __m256 red12_floats = _mm256_cvtepi32_ps(red12_ints);
        __m256 red12_results = _mm256_mul_ps(red12_floats, Scalar12);

        __m128i green12_chars = _mm_loadu_si128((__m128i*) (greens+X +1));
        __m256i green12_ints = _mm256_cvtepu8_epi32(green12_chars);
        __m256 green12_floats = _mm256_cvtepi32_ps(green12_ints);
        __m256 green12_results = _mm256_mul_ps(green12_floats, Scalar12);

        __m128i blue12_chars = _mm_loadu_si128((__m128i*) (blues+X +1));
        __m256i blue12_ints = _mm256_cvtepu8_epi32(blue12_chars);
        __m256 blue12_floats = _mm256_cvtepi32_ps(blue12_ints);
        __m256 blue12_results = _mm256_mul_ps(blue12_floats, Scalar12);

        __m128i red20_chars = _mm_loadu_si128((__m128i*) (reds+X + input_jpeg.width-1));
        __m256i red20_ints = _mm256_cvtepu8_epi32(red20_chars);
        __m256 red20_floats = _mm256_cvtepi32_ps(red20_ints);
        __m256 red20_results = _mm256_mul_ps(red20_floats, Scalar20);

        __m128i green20_chars = _mm_loadu_si128((__m128i*) (greens+X + input_jpeg.width-1));
        __m256i green20_ints = _mm256_cvtepu8_epi32(green20_chars);
        __m256 green20_floats = _mm256_cvtepi32_ps(green20_ints);
        __m256 green20_results = _mm256_mul_ps(green20_floats, Scalar20);

        __m128i blue20_chars = _mm_loadu_si128((__m128i*) (blues+X + input_jpeg.width-1));
        __m256i blue20_ints = _mm256_cvtepu8_epi32(blue20_chars);
        __m256 blue20_floats = _mm256_cvtepi32_ps(blue20_ints);
        __m256 blue20_results = _mm256_mul_ps(blue20_floats, Scalar20);

        __m128i red21_chars = _mm_loadu_si128((__m128i*) (reds+X + input_jpeg.width));
        __m256i red21_ints = _mm256_cvtepu8_epi32(red21_chars);
        __m256 red21_floats = _mm256_cvtepi32_ps(red21_ints);
        __m256 red21_results = _mm256_mul_ps(red21_floats, Scalar21);

        __m128i green21_chars = _mm_loadu_si128((__m128i*) (greens+X + input_jpeg.width));
        __m256i green21_ints = _mm256_cvtepu8_epi32(green21_chars);
        __m256 green21_floats = _mm256_cvtepi32_ps(green21_ints);
        __m256 green21_results = _mm256_mul_ps(green21_floats, Scalar21);

        __m128i blue21_chars = _mm_loadu_si128((__m128i*) (blues+X + input_jpeg.width));
        __m256i blue21_ints = _mm256_cvtepu8_epi32(blue21_chars);
        __m256 blue21_floats = _mm256_cvtepi32_ps(blue21_ints);
        __m256 blue21_results = _mm256_mul_ps(blue21_floats, Scalar21);

        __m128i red22_chars = _mm_loadu_si128((__m128i*) (reds+X + input_jpeg.width+1));
        __m256i red22_ints = _mm256_cvtepu8_epi32(red22_chars);
        __m256 red22_floats = _mm256_cvtepi32_ps(red22_ints);
        __m256 red22_results = _mm256_mul_ps(red22_floats, Scalar22);

        __m128i green22_chars = _mm_loadu_si128((__m128i*) (greens+X + input_jpeg.width+1));
        __m256i green22_ints = _mm256_cvtepu8_epi32(green22_chars);
        __m256 green22_floats = _mm256_cvtepi32_ps(green22_ints);
        __m256 green22_results = _mm256_mul_ps(green22_floats, Scalar22);

        __m128i blue22_chars = _mm_loadu_si128((__m128i*) (blues+X + input_jpeg.width+1));
        __m256i blue22_ints = _mm256_cvtepu8_epi32(blue22_chars);
        __m256 blue22_floats = _mm256_cvtepi32_ps(blue22_ints);
        __m256 blue22_results = _mm256_mul_ps(blue22_floats, Scalar22);

        //Similar to greyscale process but for 3 channels 

        __m256 sum_r = _mm256_add_ps(red00_results, 
        _mm256_add_ps(red01_results, 
        _mm256_add_ps(red02_results, 
        _mm256_add_ps(red10_results, 
        _mm256_add_ps(red11_results, 
        _mm256_add_ps(red12_results, 
        _mm256_add_ps(red20_results, 
        _mm256_add_ps(red21_results, red22_results)))))))); //too lazy to seperate lines :3

            // Convert the float32 results to int32
        __m256i sum_r_ints =  _mm256_cvtps_epi32(sum_r);
        // Seperate the 256bits result to 2 128bits result
        __m128i low_r = _mm256_castsi256_si128(sum_r_ints);
        __m128i high_r = _mm256_extracti128_si256(sum_r_ints, 1);    

        // shuffling int32s to u_int8s
        // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
        __m128i trans_low_r = _mm_shuffle_epi8(low_r, shuffle);
        __m128i trans_high_r = _mm_shuffle_epi8(high_r, shuffle);

        _mm_storeu_si128((__m128i*)(&redSmooth[X]), trans_low_r);
        _mm_storeu_si128((__m128i*)(&redSmooth[X+4]), trans_high_r);

        __m256 sum_g = _mm256_add_ps(green00_results,
         _mm256_add_ps(green01_results, 
         _mm256_add_ps(green02_results, 
         _mm256_add_ps(green10_results, 
         _mm256_add_ps(green11_results, 
         _mm256_add_ps(green12_results, 
         _mm256_add_ps(green20_results, 
         _mm256_add_ps(green21_results, green22_results)))))))); //too lazy to seperate lines :3

        __m256i sum_g_ints =  _mm256_cvtps_epi32(sum_g);

            // Seperate the 256bits result to 2 128bits result
        __m128i low_g = _mm256_castsi256_si128(sum_g_ints);
        __m128i high_g = _mm256_extracti128_si256(sum_g_ints, 1);    

        // shuffling int32s to u_int8s
        // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
        __m128i trans_low_g = _mm_shuffle_epi8(low_g, shuffle);
        __m128i trans_high_g = _mm_shuffle_epi8(high_g, shuffle);

        _mm_storeu_si128((__m128i*)(&greenSmooth[X]), trans_low_g);
        _mm_storeu_si128((__m128i*)(&greenSmooth[X+4]), trans_high_g);

        __m256 sum_b = _mm256_add_ps(blue00_results, 
        _mm256_add_ps(blue01_results, 
        _mm256_add_ps(blue02_results, 
        _mm256_add_ps(blue10_results, 
        _mm256_add_ps(blue11_results, 
        _mm256_add_ps(blue12_results, 
        _mm256_add_ps(blue20_results, 
        _mm256_add_ps(blue21_results, blue22_results)))))))); //too lazy to seperate lines :3

        __m256i sum_b_ints =  _mm256_cvtps_epi32(sum_b);

            // Seperate the 256bits result to 2 128bits result
        __m128i low_b = _mm256_castsi256_si128(sum_b_ints);
        __m128i high_b = _mm256_extracti128_si256(sum_b_ints, 1);    

        // shuffling int32s to u_int8s
        // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
        __m128i trans_low_b = _mm_shuffle_epi8(low_b, shuffle);
        __m128i trans_high_b = _mm_shuffle_epi8(high_b, shuffle);

        _mm_storeu_si128((__m128i*)(&blueSmooth[X]), trans_low_b);
        _mm_storeu_si128((__m128i*)(&blueSmooth[X+4]), trans_high_b);

    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        filteredImage[i * input_jpeg.num_channels]
                = redSmooth[i];
        filteredImage[i * input_jpeg.num_channels+1]
                = greenSmooth[i];
        filteredImage[i * input_jpeg.num_channels+2]
                = blueSmooth[i];
    }


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
