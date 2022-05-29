#pragma once

#include "read.hpp"
#include <cuda_runtime_api.h>
#include "utils/utils.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/viz/vizcore.hpp>
#include "utils/loguru.hpp"

#include "utils/cutil_math.h"
#define VOXELSIZE (0.01f)
#include <math.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include "cuVector.cuh"
#include <math_constants.h>


using namespace std;
using namespace cv;
#pragma pack(push, 1)
struct Intr
{
    union
    {
        float4 cam;
        struct
        {
            float fx, fy, cx, cy;
        };
    };
    Intr(float4 intr)
    {
        cam = intr;
    }
    __device__ inline float3 cam2world(int u, int v, float z)
    {
        // TODO 这里可预先计算fx，fy倒数，然后计算乘法，应该会快一些
        float x = __fdividef(z * (u - cx), fx);
        float y = __fdividef(z * (v - cy), fy);
        return make_float3(x, y, z);
    }
};

struct MAT3f
{
    float3 data[3];
};
inline __host__ __device__ float3 operator*(float3 &b, float3 &a)
{
    return make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
}
struct Aff3f
{
    MAT3f R;
    float3 t;
    friend inline __host__ __device__ float3 operator*(const Aff3f M, const float3 v);
};

// std::ostream &operator<<(std::ostream &out, const Matrix4 &m);
Aff3f operator*(const Aff3f &A, const Aff3f &B);
// Aff3f inverse(const Aff3f &A);

inline __host__ __device__ float3 operator*(const MAT3f M, const float3 v)
{
    return make_float3(dot(M.data[0], v), dot(M.data[1], v), dot(M.data[2], v));
}

// inline __host__ __device__ float3 operator*(const Matrix4 &M, const float3 &v)
// {
//     return make_float3(
//         dot(make_float3(M.data[0]), v) + M.data[0].w,
//         dot(make_float3(M.data[1]), v) + M.data[1].w,
//         dot(make_float3(M.data[2]), v) + M.data[2].w);
// }

// inline __host__ __device__ float3 rotate(const Matrix4 &M, const float3 &v)
// {
//     return make_float3(
//         dot(make_float3(M.data[0]), v),
//         dot(make_float3(M.data[1]), v),
//         dot(make_float3(M.data[2]), v));
// }

union Color
{
    uint8_t rgb[3];
    struct
    {
        uint8_t r, g, b;
    };
};
struct Vovel
{
    float tsdf;
    float weight;
    uchar3 color;
};

__global__ void reset(struct Vovel *vol);

struct Grid
{
    float3 center;        //中心坐标
    uint3 size;           //体素格子尺寸
    struct Vovel *m_data; //数据首指针
    float sca;
    Grid(uint3 size) : size(size)
    {
        cudaMalloc((void **)&m_data, size.x * size.y * size.z * sizeof(struct Vovel));
        reset<<<size.x, size.y>>>(m_data);
        sca = 0.01f;
    }
    __device__ inline Vovel &operator()(int x, int y, int z)
    {

        return m_data[x + y * size.x + z * size.x * size.y];
    }
    __device__ inline void set(int x, int y, int z, const struct Vovel val)
    {
        m_data[x + y * size.x + z * size.x * size.y] = val;
    }
    __device__ inline float3 getWorld(int3 pos_vol)
    {

        return pos_vol * sca + center;
    }
    //世界坐标
    __device__ inline struct Vovel fetch(const float3 &p) const
    {
        // rounding to nearest even
        float3 intpose = (p - center) / sca;
        int x = __float2int_rn(intpose.x);
        int y = __float2int_rn(intpose.y);
        int z = __float2int_rn(intpose.z);
        return m_data[x + y * size.x + z * size.x * size.y];
    }
    __device__ float interpolate(const float3 &p_voxels)
    {
        float3 cf = p_voxels;

        // // rounding to negative infinity
        int3 g = make_int3(__float2int_rd(cf.x), __float2int_rd(cf.y), __float2int_rd(cf.z));

        if (g.x < 0 || g.x >= size.x - 1 || g.y < 0 || g.y >= size.y - 1 || g.z < 0 || g.z >= size.z - 1)
            return CUDART_NAN_F;

        float a = cf.x - g.x;
        float b = cf.y - g.y;
        float c = cf.z - g.z;

        float tsdf = 0.f;
        tsdf += (this->operator()(g.x + 0, g.y + 0, g.z + 0)).tsdf * (1 - a) * (1 - b) * (1 - c);
        tsdf += (this->operator()(g.x + 0, g.y + 0, g.z + 1)).tsdf  * (1 - a) * (1 - b) * c;
        tsdf += (this->operator()(g.x + 0, g.y + 1, g.z + 0)).tsdf  * (1 - a) * b * (1 - c);
        tsdf += (this->operator()(g.x + 0, g.y + 1, g.z + 1)).tsdf * (1 - a) * b * c;
        tsdf += (this->operator()(g.x + 1, g.y + 0, g.z + 0)).tsdf * a * (1 - b) * (1 - c);
        tsdf += (this->operator()(g.x + 1, g.y + 0, g.z + 1)).tsdf * a * (1 - b) * c;
        tsdf += (this->operator()(g.x + 1, g.y + 1, g.z + 0)).tsdf * a * b * (1 - c);
        tsdf += (this->operator()(g.x + 1, g.y + 1, g.z + 1)).tsdf * a * b * c;
        return tsdf;
    }
};

#pragma pack(pop)

__global__ void depth2camkernel(Patch<uint16_t> pdepth, float3 *output, Intr intr);
__global__ void integrate(uint16_t *hd_depth_ptr, uint8_t *output, struct Vovel *vol, Intr intr, float *pose);
__global__ void exintegrate(struct Vovel *vol, float3 *output, unsigned int *num);
__global__ void reset(struct Vovel *vol, unsigned int *num);
class TSDF
{
public:
    uint3 size;
    int2 img_size;
    Intr *pintr;
    Patch<uint16_t> pdepth;
    Patch<uchar3> prgb;
    struct Grid *grid;
    TSDF(uint3 size, int2 img_size);
    void addScan(const Mat &depth, const Mat &color, cv::Affine3f pose = cv::Affine3f::Identity());
    void depth2cam(const Mat &depth, const Mat &color, Mat &depthout, Mat &colorout, cv::Affine3f pose);
    void exportCloud(Mat &cpu_cloud, Mat &cpu_color, cv::Affine3f pose = cv::Affine3f::Identity());
    void rayCast(Mat &depth, Mat &normal, cv::Affine3f pose);
    __device__ struct Vovel &getVol(int3 pose);
    ~TSDF();

private:
    // struct Vovel *DevPtr;
};
