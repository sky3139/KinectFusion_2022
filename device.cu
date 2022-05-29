
#include "device.cuh"

using namespace std;
using namespace cv;
__global__ void depth2camkernel(Patch<uint16_t> pdepth, float3 *output, Intr intr)
{
    int tx = threadIdx.x; // 640
    int ty = blockIdx.x;  // 480
    int rawidx = tx + ty * 640;
    float pz = pdepth(ty, tx) * 0.0002f; //深度转换为米
    output[rawidx] = intr.cam2world(tx, ty, pz);
}

__global__ void integrate(Patch<uint16_t> pdepth, Patch<uchar3> rgb, struct Vovel *vol, Intr intr, float *pose)
{
    int tx = threadIdx.x; // 640
    int ty = blockIdx.x;  // 480

    __shared__ float cam2base[12];
    // float *cam2base = pose;
    if (0 == tx) //同一个thread使用共享内存速度更快
    {
        for (int i = 0; i < 12; i++)
            cam2base[i] = pose[i];
    }
    __syncthreads();

    // float3 pose_t = make_float3(cam2base[0].z, cam2base[1].z, cam2base[2].z);
    int im_width = 640;
    float trunc_margin = 0.2f;
    float3 base = make_float3(1, 1, 1);
    for (int i = 0; i < 512; i++)
    {

        // 计算小体素的世界坐标weight_old

        float3 vol_world = make_float3(i - 100, tx - 100, ty - 100) * VOXELSIZE;

        // //     //计算体素在相机坐标系的坐标
        float tmp_pt[3] = {0};
        tmp_pt[0] = vol_world.x - cam2base[0 * 4 + 3];
        tmp_pt[1] = vol_world.y - cam2base[1 * 4 + 3];
        tmp_pt[2] = vol_world.z - cam2base[2 * 4 + 3];
        float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

        if (pt_cam_z <= 0)
            continue;

        int pt_pix_x = roundf(intr.fx * (pt_cam_x / pt_cam_z) + intr.cx);
        int pt_pix_y = roundf(intr.fy * (pt_cam_y / pt_cam_z) + intr.cy);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= 480)
            continue;

        uint16_t img_depu2 = pdepth(pt_pix_x, pt_pix_y); //
        float img_dep = img_depu2 * 0.0002f;
        if (img_dep <= 0 || img_dep > 6)
            continue;
        float diff = img_dep - pt_cam_z;
        if (diff <= -trunc_margin)
            continue;
        int idx = i + tx * 512 + ty * 512 * 512;
        struct Vovel &p = vol[idx];
        const float dist = fminf(1.0f, __fdividef(diff, trunc_margin));

        float weight_old = (float)p.weight; 
        p.weight = p.weight > 250 ? p.weight : p.weight + 1;
        float weight_new = (float)p.weight;
        p.tsdf = (p.tsdf * weight_old + dist) / weight_new;


        uchar3 rgb_val = rgb(pt_pix_x, pt_pix_y); // pt->rgb[0];
        uint16_t mval = (p.color.x * weight_old + rgb_val.x) / weight_new;
        p.color.x = mval > 255 ? 255 : mval;
        mval = (p.color.y * weight_old + rgb_val.y) / weight_new;
        p.color.y = mval > 255 ? 255 : mval;
        mval = (p.color.z * weight_old + rgb_val.z) / weight_new;
        p.color.z = mval > 255 ? 255 : mval;

    }
}
__global__ void exintegrate(struct Vovel *vol, float3 *output, uchar3 *rgb, unsigned int *num)
{
    int vx = threadIdx.x; // 640
    int ty = blockIdx.x;  // 480

    for (int vz = 0; vz < 512; vz++)
    {
        int idx = vz + vx * 512 + ty * 512 * 512;
        const struct Vovel &p = vol[idx];
        if ((vx == 0 && ty == 0) || (vz == 0 && ty == 0) || (vz == 0 && vx == 0))
        {
            unsigned int val = atomicInc(num, 0xffffff);
            output[val] = make_float3(vx - 100, vz - 100, ty - 100) * VOXELSIZE;
            rgb[val] = p.color;
        }
        if ((p.weight > 0) && fabs(p.tsdf) < 0.20f)
        // if (tx % 5 == 0 && ty % 5 == 0 & vz % 5 == 0)
        {
            unsigned int val = atomicInc(num, 0xffffff);
            output[val] = make_float3(vx - 100, vz - 100, ty - 100) * VOXELSIZE;
            rgb[val] = p.color;
        }
    }
}
__global__ void reset(struct Vovel *vol)
{
    int tx = threadIdx.x; // 640
    int ty = blockIdx.x;  // 480

    for (int vz = 0; vz < 512; vz++)
    {
        int idx = vz + tx * 512 + ty * 512 * 512;
        vol[idx].weight = 0.0f;
        vol[idx].tsdf = 1.0f;
        vol[idx].color = make_uchar3(0, 0, 0);
    }
}

TSDF::TSDF(float3 size, int2 img_size) : size(size), img_size(img_size)
{
    cudaMalloc((void **)&DevPtr, size.x * size.y * size.z * sizeof(struct Vovel));
    ck(cudaGetLastError());
    struct Vovel v;
    v.tsdf = 0.0f;
    v.weight = 1.0f;

    pdepth.creat(640, 480);
    prgb.creat(640, 480);

    reset<<<size.x, size.y>>>(DevPtr);
    ck(cudaDeviceSynchronize());
}

void TSDF::depth2cam(const Mat &depth_in, const Mat &color_in, Mat &cloud_out, Mat &color_out, cv::Affine3f pose)
{
    pdepth.upload(depth_in.ptr<uint16_t>(), depth_in.step);

    float3 *d_cloud;
    cudaMallocManaged(&d_cloud, depth_in.rows * depth_in.cols * sizeof(float3));
    depth2camkernel<<<depth_in.rows, depth_in.cols>>>(pdepth, d_cloud, *pintr);
    ck(cudaDeviceSynchronize());
    cloud_out = cv::Mat(depth_in.rows * depth_in.cols, 1, CV_32FC3);
    cudaMemcpy(cloud_out.ptr<float>(), d_cloud, depth_in.rows * depth_in.cols * sizeof(float3), cudaMemcpyDeviceToHost);
    color_out = color_in.reshape(3, color_in.rows * color_in.cols);
    cudaFree(d_cloud);
}
void TSDF::addScan(const Mat &depth, const Mat &color, cv::Affine3f pose)
{

    pdepth.upload(depth.ptr<uint16_t>(), depth.step);
    prgb.upload(color.ptr<uchar3>(), color.step);

    ck(cudaGetLastError());

    float *hd_pose;
    cudaMalloc((void **)&hd_pose, 16 * sizeof(float));
    ck(cudaGetLastError());

    cudaMemcpy(hd_pose, pose.matrix.val, 16 * sizeof(float), cudaMemcpyHostToDevice);
    ck(cudaGetLastError());

    integrate<<<size.x, size.y>>>(pdepth, prgb, DevPtr, *pintr, hd_pose);
    ck(cudaDeviceSynchronize());
    cudaFree(hd_pose);
}

void TSDF::exportCloud(std::shared_ptr<Mat> &cpu_cloud, Mat &cpu_color, cv::Affine3f pose)
{
    cpu_cloud = std::make_shared<Mat>();
    cpu_color.create(img_size.x * img_size.y, 1, CV_8UC3); 

    dim3 grid(size.x, size.y, size.z);
    dim3 block(1, 1, 1);
    float3 *output;
    unsigned int *num;
    cudaMallocManaged((void **)&num, sizeof(unsigned int));
    size_t all_num = size.x * size.y * size.z / 3;
    *num = 0;
    cudaMallocManaged((void **)&output, sizeof(float3) * all_num);

    uchar3 *d_color;
    cudaMallocManaged((void **)&d_color, sizeof(uchar3) * all_num);

    exintegrate<<<size.x, size.y>>>(DevPtr, output, d_color, num);
    ck(cudaDeviceSynchronize());

    cout << *num << " " << all_num << endl;
    cpu_cloud = std::make_shared<Mat>(img_size.x * img_size.y, 1, CV_32FC3);
    cudaMemcpy((float *)cpu_cloud->data, &output[0].x, sizeof(float3) * (*num), cudaMemcpyDeviceToHost);
    cudaMemcpy((uint8_t *)cpu_color.data, &d_color[0].x, sizeof(uint8_t) * 3 * (*num), cudaMemcpyDeviceToHost);

    cudaFree(output);
    cudaFree(num);
}
__device__ struct Vovel &TSDF::getVol(int3 pose)
{
    int idx = pose.x + pose.y * size.x + pose.z * size.x * size.y;
    return (DevPtr[idx]);
}
TSDF::~TSDF()
{
    pdepth.release();
    prgb.release();
    cudaFree(DevPtr);
}
