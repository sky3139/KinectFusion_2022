#pragma once

#include "read.hpp"
#include <cuda_runtime_api.h>
#include "utils/utils.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/viz/vizcore.hpp>

#include "utils/cutil_math.h"
#define VOXELSIZE (0.01f)
#include <math.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include "cuVector.cuh"
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
#pragma pack(pop)

__global__ void depth2camkernel(Patch<uint16_t> pdepth, float3 *output, Intr intr);
__global__ void integrate(uint16_t *hd_depth_ptr, uint8_t *output, struct Vovel *vol, Intr intr, float *pose);
__global__ void exintegrate(struct Vovel *vol, float3 *output, unsigned int *num);
__global__ void reset(struct Vovel *vol, unsigned int *num);
class TSDF
{
public:
    float3 size;
    int2 img_size;
    Intr *pintr;
    Patch<uint16_t> pdepth;
    Patch<uchar3> prgb;
    TSDF(float3 size, int2 img_size);
    void addScan(const Mat &depth, const Mat &color, cv::Affine3f pose = cv::Affine3f::Identity());
    void depth2cam(const Mat &depth, const Mat &color, Mat &depthout, Mat &colorout, cv::Affine3f pose);
    void exportCloud(std::shared_ptr<Mat> &cpu_cloud, Mat &cpu_color, cv::Affine3f pose = cv::Affine3f::Identity());
    __device__ struct Vovel &getVol(int3 pose);
    ~TSDF();

private:
    struct Vovel *DevPtr;
};
