
#include "TSDF.cuh"
#include "RayCast.cuh"

using namespace std;
using namespace cv;
__global__ void depth2camkernel(Patch<uint16_t> pdepth, float3 *output, Intr intr)
{
    int tx = threadIdx.x; // 640
    int vy = blockIdx.x;  // 480
    int rawidx = tx + vy * 640;
    float pz = pdepth(vy, tx) * 0.0002f; //深度转换为米
    output[rawidx] = intr.cam2world(tx, vy, pz);
}

__global__ void integrate(cudaTextureObject_t obj, Patch<uint16_t> pdepth, Patch<uchar3> rgb, uint8_t *rgbda, struct Grid grid, Intr intr, float *pose)
{
    int tx = threadIdx.x; // 640
    int vy = blockIdx.x;  // 480

    __shared__ float cam2base[12];
    // float *cam2base = pose;
    if (0 == tx) //同一个thread使用共享内存速度更快
    {
        for (int i = 0; i < 12; i++)
            cam2base[i] = pose[i];
    }

    // float3 pose_t = make_float3(cam2base[0].z, cam2base[1].z, cam2base[2].z);
    int im_width = 640;
    float trunc_margin = grid.sca * 5;
    float3 base = make_float3(1, 1, 1);
    __syncthreads();
    for (int i = 0; i < grid.size.z; i++)
    {
        // 计算小体素的世界坐标weight_old

        float3 vol_world = grid.getWorld(make_int3(i, tx, vy)); // make_float3(i - 100, tx - 100, vy - 100) * VOXELSIZE;

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

        int pt_pix_x = rintf(intr.fx * (pt_cam_x / pt_cam_z) + intr.cx);
        int pt_pix_y = rintf(intr.fy * (pt_cam_y / pt_cam_z) + intr.cy);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= 480)
            continue;

        // uint16_t img_depu2 = ppdepth[pt_pix_y * 640 + pt_pix_x]; //
        // uint16_t img_depu2 = pdepth(pt_pix_y, pt_pix_x); //
        uint16_t img_depu2 = tex2D<uint16_t>(obj,pt_pix_x, pt_pix_y); //

        float img_dep = img_depu2 * 0.0002f;
        if (img_dep <= 0 || img_dep > 6)
            continue;
        float diff = img_dep - pt_cam_z;
        if (diff <= -trunc_margin)
            continue;
        struct Vovel &p = (grid)(i, tx, vy); //[idx];
        const float dist = fminf(1.0f, __fdividef(diff, trunc_margin));
        float weight_old = (float)p.weight;
        p.weight = p.weight > 250 ? p.weight : p.weight + 1;
        float weight_new = (float)p.weight;
        p.tsdf = (p.tsdf * weight_old + dist) / weight_new;
        uchar3 rgb_val;

        rgb_val.x = rgbda[(pt_pix_y * 640 + pt_pix_x * 1) * 3];
        rgb_val.z = rgbda[(pt_pix_y * 640 + pt_pix_x * 1) * 3 + 1];
        rgb_val.y = rgbda[(pt_pix_y * 640 + pt_pix_x * 1) * 3 + 2];

        uint16_t mval = (p.color.x * weight_old + rgb_val.x) / weight_new;
        p.color.x = mval > 255 ? 255 : mval;
        mval = (p.color.y * weight_old + rgb_val.y) / weight_new;
        p.color.y = mval > 255 ? 255 : mval;
        mval = (p.color.z * weight_old + rgb_val.z) / weight_new;
        p.color.z = mval > 255 ? 255 : mval;
    }
}
__global__ void exintegrate(struct Grid grid, float3 *output, uchar3 *rgb, unsigned int *num)
{
    int vx = threadIdx.x; // 640
    int vy = blockIdx.x;  // 480
    if (vx >= blockDim.x || vy >= blockDim.x)
        return;
    for (int vz = 0; vz < grid.size.z; vz++)
    {
        struct Vovel &p = (grid)(vz, vx, vy); //[idx];

        if ((vx == 0 && vy == 0) || (vz == 0 && vy == 0) || (vz == 0 && vx == 0))
        {
            unsigned int val = atomicInc(num, 0xffffff);
            output[val] = grid.getWorld(make_int3(vz, vx, vy));
            rgb[val] = p.color;
        }
        else if ((p.weight > 0.90) && fabs(p.tsdf) < 0.20f)
        // if (tx % 5 == 0 && vy % 5 == 0 & vz % 5 == 0)
        {
            unsigned int val = atomicInc(num, 0xffffff);
            output[val] = grid.getWorld(make_int3(vz, vx, vy));
            rgb[val] = p.color;
        }
    }
}
__global__ void reset(struct _Vovel *vol)
{
    int tx = threadIdx.x; // 640
    int vy = blockIdx.x;  // 480
    for (int vz = 0; vz < 512; vz++)
    {
        // int idx = vz + tx * 512 + vy * 512 * 512;
        vol->m_data[tx][vy][vz].weight = 0.0f;
        vol->m_data[tx][vy][vz].tsdf = 1.0f;
        vol->m_data[tx][vy][vz].color = make_uchar3(255, 255, 255);
    }
}

TSDF::TSDF(uint3 size, int2 img_size) : size(size), img_size(img_size)
{
    ck(cudaGetLastError());
    struct Vovel v;
    v.tsdf = 0.0f;
    v.weight = 1.0f;

    pdepth.creat(480, 640);
    prgb.creat(480, 640);
    grid = new Grid(size);
    grid->center = make_float3(-3, -2, 0);
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
#include "TextureBinder.hpp"
void TSDF::addScan(const Mat &depth, const Mat &color, cv::Affine3f pose)
{

    pdepth.upload(depth.ptr<uint16_t>(), depth.step);
    prgb.upload(color.ptr<uchar3>(), color.step);

    Cuda::TextureBinder tb(pdepth);
    // cv::transpose(color,cp);
    // img1 = cv.transpose(img);
    // prgb.upload(cp.ptr<uchar>(), 480 * 3);
    uint8_t *rgbda;
    ck(cudaMallocManaged((void **)&rgbda, 640 * 480 * 3));
    memcpy(rgbda, color.ptr<uchar>(), 640 * 480 * 3);
    // ck(cudaMemcpy2D(prgb.devPtr, prgb.pitch, (void *)color.ptr<uchar3>(), 640 * 3, sizeof(uchar3) * prgb.cols, prgb.rows, cudaMemcpyHostToDevice));
    // prgb.download(cp.ptr<uchar3>(), cp.step);
    // cv::imshow("a", cp);
    // cv::waitKey(100);
    // prgb.print();
    // cout << color.step << " " << sizeof(uchar3) << endl;
    ck(cudaGetLastError());

    float *hd_pose;
    cudaMalloc((void **)&hd_pose, 16 * sizeof(float));
    ck(cudaGetLastError());
    // cout << pose.matrix << endl;
    cudaMemcpy(hd_pose, pose.matrix.val, 16 * sizeof(float), cudaMemcpyHostToDevice);
    ck(cudaGetLastError());

    integrate<<<size.x, size.y>>>(tb.obj, pdepth, prgb, rgbda, *grid, *pintr, hd_pose);
    ck(cudaDeviceSynchronize());
    cudaFree(hd_pose);
    cudaFree(rgbda);
}

void TSDF::exportCloud(Mat &cpu_cloud, Mat &cpu_color, cv::Affine3f pose)
{
    // cpu_cloud.create(img_size.x * img_size.y, 1, CV_32FC3);
    // cpu_color.create(img_size.x * img_size.y, 1, CV_8UC3);
    cpu_cloud = Mat();
    cpu_color = Mat();
    float3 *output;
    unsigned int *num;
    ck(cudaMallocManaged((void **)&num, sizeof(unsigned int)));
    size_t all_num = size.x * size.y * size.z / 5;
    ck(cudaMemset(num, 0x00, sizeof(unsigned int)));
    ck(cudaMallocManaged((void **)&output, sizeof(float3) * all_num));

    uchar3 *d_color;
    ck(cudaMallocManaged((void **)&d_color, sizeof(uchar3) * all_num));

    exintegrate<<<size.x, size.y>>>(*grid, output, d_color, num);
    ck(cudaDeviceSynchronize());

    // cout << *num << " " << all_num << endl;
    for (int i = 0; i < *num; i++)
    {
        cpu_cloud.push_back(cv::Vec3f(output[i].x, output[i].y, output[i].z));
        cpu_color.push_back(cv::Vec3b(d_color[i].x, d_color[i].y, d_color[i].z));
    }
    // (memcpy(cpu_cloud.ptr<float3>(), output, sizeof(float3) * (*num)));
    // cudaMemcpy((uint8_t *)cpu_color.data, &d_color[0].x, sizeof(uchar3) * (*num), cudaMemcpyDeviceToHost);

    cudaFree(output);
    cudaFree(num);
    cudaFree(d_color);
}

void TSDF::rayCast(Mat &depth, Mat &normal, cv::Affine3f camera_pose)
{
    Patch<uint16_t> _pdepth(img_size.y, img_size.x);

    Patch<float4> pnormal(img_size.x, img_size.y);
    depth.create(img_size.y, img_size.x, CV_16UC1);

    auto pose_ = cv::Affine3f().translate(Vec3f(-2.5 / 2, -2.5 / 2, -2.5 / 2));

    float *hd_pose;
    cudaMalloc((void **)&hd_pose, 16 * sizeof(float));
    ck(cudaGetLastError());
    // cout << camera_pose.matrix << endl;
    cudaMemcpy(hd_pose, camera_pose.matrix.val, 16 * sizeof(float), cudaMemcpyHostToDevice);
    ck(cudaGetLastError());

    cv::Affine3f cam2vol = pose_.inv() * camera_pose;
    device::Aff3f aff = device::device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device::device_cast<device::Mat3f>(cam2vol.rotation().inv(cv::DECOMP_SVD));

    cv::Vec3i dims_(256, 256, 256);
    cv::Vec3f vsize(0.02, 0.02, 0.02);

    int3 dims = make_int3(256, 256, 256);
    float3 vsz = make_float3(0.02, 0.02, 0.02);

    // device::TsdfVolume volume(data_.ptr(), dims, vsz, trunc_dist_, max_weight_);
    // device::raycast(volume, aff, Rinv, reproj, p, n, raycast_step_factor_, gradient_delta_factor_);

    int threads_per_block = 64;
    int thread_blocks = (640 * 480 + threads_per_block - 1) / threads_per_block;
    raycast_kernel<<<size.x, size.y>>>(_pdepth, pnormal, *grid, *pintr, aff, Rinv, hd_pose);

    // raycast_kernel<<<thread_blocks, threads_per_block>>>(pdepth, pnormal, *grid, *pintr, aff, Rinv);
    ck(cudaGetLastError());

    // // // cpu_cloud.create(img_size.x * img_size.y, 1, CV_32FC3);
    // normal.create(img_size.y, img_size.x, CV_32FC4);
    _pdepth.download(depth.ptr<uint16_t>(), depth.step);

    cv::imshow("tsdfdepth", depth);
    _pdepth.release();
    normal.release();
    // pnormal.download(depth.ptr<float3>(), normal.step);

    // cpu_cloud = Mat();
    // cpu_color = Mat();
    // float3 *output;
    // unsigned int *num;
    // ck(cudaMallocManaged((void **)&num, sizeof(unsigned int)));
    // size_t all_num = size.x * size.y * size.z / 5;
    // ck(cudaMemset(num, 0x00, sizeof(unsigned int)));
    // ck(cudaMallocManaged((void **)&output, sizeof(float3) * all_num));

    // uchar3 *d_color;
    // ck(cudaMallocManaged((void **)&d_color, sizeof(uchar3) * all_num));

    // exintegrate<<<size.x, size.y>>>(*grid, output, d_color, num);
    // ck(cudaDeviceSynchronize());

    // cout << *num << " " << all_num << endl;
    // for (int i = 0; i < *num; i++)
    // {
    //     cpu_cloud.push_back(cv::Vec3f(output[i].x, output[i].y, output[i].z));
    //     cpu_color.push_back(cv::Vec3b(d_color[i].x, d_color[i].y, d_color[i].z));
    // }
    // // (memcpy(cpu_cloud.ptr<float3>(), output, sizeof(float3) * (*num)));
    // // cudaMemcpy((uint8_t *)cpu_color.data, &d_color[0].x, sizeof(uchar3) * (*num), cudaMemcpyDeviceToHost);

    // cudaFree(output);
    // cudaFree(num);
    cudaFree(hd_pose);
}
// __device__ struct Vovel &TSDF::getVol(int3 pose)
// {
//     int idx = pose.x + pose.y * size.x + pose.z * size.x * size.y;
//     return (DevPtr[idx]);
// }
TSDF::~TSDF()
{
    pdepth.release();
    prgb.release();
    // cudaFree(DevPtr);
}
