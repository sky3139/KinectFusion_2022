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
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float3 operator*(int3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(uint3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator-(const float3 &b,const float3 &a)
{
    return make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
}
inline __host__ __device__ float3 operator+(const float3 &b, const float3 &a)
{
    return make_float3(b.x + a.x, b.y + a.y, b.z + a.z);
}
inline __host__ __device__ float3 operator+=(float3 &b, const float3 &a)
{
    b = make_float3(b.x + a.x, b.y + a.y, b.z + a.z);
}
inline __host__ __device__ float operator*(float4 &b, float3 &a)
{
    return (b.x * a.x + b.y * a.y + b.z * a.z);
}
inline __host__ __device__ float3 operator*(const float3 &v1, const float3 &v2)
{
    return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline __host__ __device__ float3 normalized(const float3 &v)
{
    return v * rsqrt(dot(v, v));
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