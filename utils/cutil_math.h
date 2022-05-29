#pragma once
#include <cuda_runtime.h>

inline __host__ __device__ float dot(float3 p1, float3 p2)
{
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator-(float3 &b, float3 &a)
{
    return make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
}
inline __host__ __device__ float operator*(float4 &b, float3 &a)
{
    return (b.x * a.x + b.y * a.y + b.z * a.z);
}

// #ifndef __CUDACC__

// ////////////////////////////////////////////////////////////////////////////////
// // host implementations of CUDA functions
// ////////////////////////////////////////////////////////////////////////////////

// inline float fminf(float a, float b)
// {
//     return a < b ? a : b;
// }

// inline float fmaxf(float a, float b)
// {
//     return a > b ? a : b;
// }

// inline int max(int a, int b)
// {
//     return a > b ? a : b;
// }

// inline int min(int a, int b)
// {
//     return a < b ? a : b;
// }

// inline float rsqrtf(float x)
// {
//     return 1.0f / sqrtf(x);
// }
// // #endif
// //