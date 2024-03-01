//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    // Read JPEG File
    const char * input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Divide the task
    // For example, there are 11 pixels and 3 tasks, 
    // we try to divide to 4 4 3 instead of 3 3 5
    int total_pixel_num = input_jpeg.width * input_jpeg.height;
    int pixel_num_per_task = total_pixel_num / numtasks;    
    int left_pixel_num = total_pixel_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_pixel_num = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }

    // The tasks for the master executor
    // 1. Transform the first division of the RGB contents to the Gray contents
    // 2. Receive the transformed Gray contents from slave executors
    // 3. Write the Gray contents to the JPEG File
    if (taskid == MASTER) {
        // Transform the first division of RGB Contents to the gray contents
        auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
        for (int i = cuts[MASTER]; i < cuts[MASTER + 1]; i++) {
            int height = i / input_jpeg.width;
            int width = i % input_jpeg.width;

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
                        
            filteredImage[i * input_jpeg.num_channels] 
                = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[i * input_jpeg.num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[i * input_jpeg.num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));

        }

        // Receive the transformed contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filteredImage + (cuts[i] * input_jpeg.num_channels);
            int length = cuts[i+1] - cuts[i];
            MPI_Recv(start_pos, length * input_jpeg.num_channels, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        

        // Save
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                             input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        // Release the memory
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } 
    // The tasks for the slave executor
    // 1. Transform the RGB contents to the Gray contents
    // 2. Send the transformed Gray contents back to the master executor
    else {
        int length = cuts[taskid + 1] - cuts[taskid]; 
        auto filteredImage = new unsigned char[length * input_jpeg.num_channels];
        for (int i = cuts[taskid]; i < cuts[taskid + 1]; i++) {

            int height = i / input_jpeg.width;
            int width = i % input_jpeg.width;

            //0 0
            int channel_value_r1 = input_jpeg.buffer[((height-1) * input_jpeg.width + (width-1)) * input_jpeg.num_channels];
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

            int j = i-cuts[taskid];

            filteredImage[j * input_jpeg.num_channels] =
                static_cast<unsigned char>(std::round(sum_r));
            filteredImage[j * input_jpeg.num_channels + 1] =
                static_cast<unsigned char>(std::round(sum_g));
            filteredImage[j * input_jpeg.num_channels + 2] =
                static_cast<unsigned char>(std::round(sum_b));

        }

        // Send the gray image back to the master
        MPI_Send(filteredImage, length * input_jpeg.num_channels, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        
        // Release the memory
        delete[] filteredImage;
    }

    MPI_Finalize();
    return 0;
}
