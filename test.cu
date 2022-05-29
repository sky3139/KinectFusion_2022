#include <iostream>
#include <stdio.h>
#include "./utils/utils.cuh"
typedef float PixelType;

__global__ void extract_patches_from_image_data(cudaPitchedPtr devicePitchedPointer, dim3 image_dimensions)
{
    // Test
    printf("HELLO - PLEASE PRINT THIS\n");

    // Check image dimensions
    printf("Current x: %d\n", image_dimensions.x);
    printf("Current y: %d\n", image_dimensions.y);
    printf("Current z: %d\n", image_dimensions.z);

    // Get attributes from device pitched pointer
    char *devicePointer = (char *)devicePitchedPointer.ptr;
    size_t pitch = devicePitchedPointer.pitch;
    size_t slicePitch = pitch * image_dimensions.y;

    // Loop over image data
    // for (int z = 0; z < image_dimensions.z; ++z)
    // {
    //     char *current_slice = devicePointer + z * slicePitch;

    //     for (int y = 0; y < image_dimensions.y; ++y)
    //     {
    //         PixelType *current_row = (PixelType *)(current_slice + y * pitch);

    //         for (int x = 0; x < image_dimensions.x; ++x)
    //         {
    //             PixelType current_element = current_row[x];

    //             printf("Current element: %d\n", current_element);
    //         }
    //     }
    // }
}

int main(void)
{
    // Set up test data
    PixelType image_data[3][3][3] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
    dim3 image_dimensions = dim3(32, 32, 32);

    // Allocate 3D memory on the device
    cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(PixelType) * image_dimensions.x, image_dimensions.y, image_dimensions.z);
    cudaPitchedPtr devicePitchedPointer;
    cudaMalloc3D(&devicePitchedPointer, volumeSizeBytes);
    ck(cudaGetLastError());

    // Kernel Launch Configuration
    dim3 threads_per_block = dim3(1, 1, 1);
    dim3 blocks_per_grid = dim3(1, 1, 1);
    extract_patches_from_image_data<<<blocks_per_grid, threads_per_block>>>(devicePitchedPointer, image_dimensions);
    ck(cudaGetLastError());
    getchar();
}
