
#pragma once
#include <cuda_runtime_api.h>
#define ck(val)                                                                                \
    {                                                                                          \
        if (val != cudaSuccess)                                                                \
        {                                                                                      \
            printf("Error ：%s:%d , ", __FILE__, __LINE__);                                    \
            printf("code : %d , reason : %s \n", cudaGetLastError(), cudaGetErrorString(val)); \
            exit(-1);                                                                          \
        }                                                                                      \
    }
