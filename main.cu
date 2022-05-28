#include "read.hpp"
#include <cuda_runtime_api.h>
#include "utils/utils.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/viz/vizcore.hpp>

using namespace std;
using namespace cv;

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
__global__ void depth2cam(uint16_t *hd_depth_ptr, float3 *output, Intr intr)
{
    int tx = threadIdx.x; // 640
    int ty = blockIdx.x;  // 480
    int rawidx = tx + ty * 640;
    float pz = hd_depth_ptr[rawidx] * 0.0002f; //深度转换为米
    output[rawidx] = intr.cam2world(tx, ty, pz);
}
class TSDF
{
public:
    TSDF()
    {
    }
    void input(const Mat &depth,const Mat &color)
    {
    }
    void integrate()
    {
    }
    void exportCloud(Mat &depth, Mat &color)
    {
    }
};
int main()
{
    DataSet<float> dt("/home/lei/dataset/paper/f3_long_office");
    ck(cudaGetLastError());
    Intr intr(make_float4(550, 550, 320, 240));
    cv::viz::Viz3d window("map");
    window.showWidget("Coordinate", cv::viz::WCoordinateSystem());
    TSDF tsdf;
    for (int i = 0; i < dt.pose.frames; i++)
    {
        cv::Mat rgb = cv::imread(dt.color_path[i]);
        cv::Mat depth = cv::imread(dt.depth_path[i], cv::IMREAD_ANYDEPTH);

        uint16_t *hd_depth_ptr;
        cudaMallocManaged((void **)&hd_depth_ptr, depth.rows * depth.cols * sizeof(uint16_t));
        cudaMemcpy(hd_depth_ptr, depth.ptr<uint16_t>(), depth.rows * depth.cols * sizeof(uint16_t), cudaMemcpyHostToDevice);
        float3 *cloud;
        cudaMallocManaged(&cloud, depth.rows * depth.cols * sizeof(float3));
        depth2cam<<<depth.rows, depth.cols>>>(hd_depth_ptr, cloud, intr);
        ck(cudaDeviceSynchronize());
        cv::Mat cpu_cloud(depth.rows * depth.cols, 1, CV_32FC3);
        cudaMemcpy(cpu_cloud.ptr<float *>(), cloud, depth.rows * depth.cols * sizeof(float3), cudaMemcpyDeviceToHost);
        cv::Mat dst2 = rgb.reshape(3, 640 * 480);
        window.showWidget("depthmode", cv::viz::WCloud(cpu_cloud, dst2));
        window.spinOnce(true);

        cv::imshow("rgb", rgb);
        cv::imshow("depth", depth);
        cv::waitKey(100);
        window.spin();
    }
}
// cv::Affine3f pose = cv::Affine3f::Identity(); //位姿先用单位矩阵
// float *hd_pose_ptr;
// cudaMallocManaged((void **)&hd_pose_ptr, 16 * sizeof(float));
// cudaMemcpy(hd_pose_ptr, pose.matrix.val, 16 * sizeof(float), cudaMemcpyHostToDevice);
// dim3 grid(1, 1, 1), block(depth.rows, depth.cols, 1); //每个block xy 方向最大thread数量为 1024
