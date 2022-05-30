#include "RayCast.cuh"
#include "utils/cutil_math.h"
// struct TsdfRaycaster
// {
//     TsdfVolume volume;

//     Aff3f aff;
//     Mat3f Rinv;

//     Vec3f volume_size;
//     Reprojector reproj;
//     float time_step;
//     float3 gradient_delta;
//     float3 voxel_size_inv;

//     TsdfRaycaster(const TsdfVolume &volume, const Aff3f &aff, const Mat3f &Rinv, const Reprojector &_reproj);

//     __device__ float fetch_tsdf(const float3 &p) const
//     {
//         // rounding to nearest even
//         int x = __float2int_rn(p.x * voxel_size_inv.x);
//         int y = __float2int_rn(p.y * voxel_size_inv.y);
//         int z = __float2int_rn(p.z * voxel_size_inv.z);
//         return unpack_tsdf(*volume(x, y, z));
//     }

//     __device__ void operator()(PtrStepSz<ushort> depth, PtrStep<Normal> normals) const
//     {
//         int x = blockIdx.x * blockDim.x + threadIdx.x;
//         int y = blockIdx.y * blockDim.y + threadIdx.y;

//         if (x >= depth.cols || y >= depth.rows)
//             return;

//         const float qnan = numeric_limits<float>::quiet_NaN();

//         depth(y, x) = 0;
//         normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

//         float3 ray_org = aff.t;
//         float3 ray_dir = normalized(aff.R * reproj(x, y, 1.f));

//         // We do subtract voxel size to minimize checks after
//         // Note: origin of volume coordinate is placed
//         // in the center of voxel (0,0,0), not in the corner of the voxel!
//         float3 box_max = volume_size - volume.voxel_size;

//         float tmin, tmax;
//         intersect(ray_org, ray_dir, box_max, tmin, tmax);

//         const float min_dist = 0.f;
//         tmin = fmax(min_dist, tmin);
//         if (tmin >= tmax)
//             return;

//         tmax -= time_step;
//         float3 vstep = ray_dir * time_step;
//         float3 next = ray_org + ray_dir * tmin;

//         float tsdf_next = fetch_tsdf(next);
//         for (float tcurr = tmin; tcurr < tmax; tcurr += time_step)
//         {
//             float tsdf_curr = tsdf_next;
//             float3 curr = next;
//             next += vstep;

//             tsdf_next = fetch_tsdf(next);
//             if (tsdf_curr < 0.f && tsdf_next > 0.f)
//                 break;

//             if (tsdf_curr > 0.f && tsdf_next < 0.f)
//             {
//                 float Ft = interpolate(volume, curr * voxel_size_inv);
//                 float Ftdt = interpolate(volume, next * voxel_size_inv);

//                 float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);

//                 float3 vertex = ray_org + ray_dir * Ts;
//                 float3 normal = compute_normal(vertex);

//                 if (!isnan(normal.x * normal.y * normal.z))
//                 {
//                     normal = Rinv * normal;
//                     vertex = Rinv * (vertex - aff.t);

//                     normals(y, x) = make_float4(normal.x, normal.y, normal.z, 0);
//                     depth(y, x) = static_cast<ushort>(vertex.z * 1000);
//                 }
//                 break;
//             }
//         } /* for (;;) */
//     }

//     __device__ float3 compute_normal(const float3 &p) const
//     {
//         float3 n;

//         float Fx1 = interpolate(volume, make_float3(p.x + gradient_delta.x, p.y, p.z) * voxel_size_inv);
//         float Fx2 = interpolate(volume, make_float3(p.x - gradient_delta.x, p.y, p.z) * voxel_size_inv);
//         n.x = __fdividef(Fx1 - Fx2, gradient_delta.x);

//         float Fy1 = interpolate(volume, make_float3(p.x, p.y + gradient_delta.y, p.z) * voxel_size_inv);
//         float Fy2 = interpolate(volume, make_float3(p.x, p.y - gradient_delta.y, p.z) * voxel_size_inv);
//         n.y = __fdividef(Fy1 - Fy2, gradient_delta.y);

//         float Fz1 = interpolate(volume, make_float3(p.x, p.y, p.z + gradient_delta.z) * voxel_size_inv);
//         float Fz2 = interpolate(volume, make_float3(p.x, p.y, p.z - gradient_delta.z) * voxel_size_inv);
//         n.z = __fdividef(Fz1 - Fz2, gradient_delta.z);

//         return normalized(n);
//     }
// };

// inline TsdfRaycaster::TsdfRaycaster(const TsdfVolume &_volume, const Aff3f &_aff, const Mat3f &_Rinv, const Reprojector &_reproj)
//     : volume(_volume), aff(_aff), Rinv(_Rinv), reproj(_reproj) {}

// __global__ void raycast_kernel(const TsdfRaycaster raycaster, PtrStepSz<ushort> depth, PtrStep<Normal> normals)
// {
//     raycaster(depth, normals);
// };

__device__ inline void intersect(float3 ray_org, float3 ray_dir, /*float3 box_min,*/ float3 box_max, float &tnear, float &tfar)
{
    float3 box_min = make_float3(0.f, 0.f, 0.f);

    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.f / ray_dir.x, 1.f / ray_dir.y, 1.f / ray_dir.z);
    float3 tbot = invR * (box_min - ray_org);
    float3 ttop = invR * (box_max - ray_org);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = make_float3(fminf(ttop.x, tbot.x), fminf(ttop.y, tbot.y), fminf(ttop.z, tbot.z));
    float3 tmax = make_float3(fmaxf(ttop.x, tbot.x), fmaxf(ttop.y, tbot.y), fmaxf(ttop.z, tbot.z));

    // find the largest tmin and the smallest tmax
    tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
}

__device__ float3 compute_normal(const struct Grid &grid, const float3 &p)
{
    float3 n;

    // float Fx1 = interpolate(volume, make_float3(p.x + gradient_delta.x, p.y, p.z) * voxel_size_inv);
    // float Fx2 = interpolate(volume, make_float3(p.x - gradient_delta.x, p.y, p.z) * voxel_size_inv);
    // n.x = __fdividef(Fx1 - Fx2, gradient_delta.x);

    // float Fy1 = interpolate(volume, make_float3(p.x, p.y + gradient_delta.y, p.z) * voxel_size_inv);
    // float Fy2 = interpolate(volume, make_float3(p.x, p.y - gradient_delta.y, p.z) * voxel_size_inv);
    // n.y = __fdividef(Fy1 - Fy2, gradient_delta.y);

    // float Fz1 = interpolate(volume, make_float3(p.x, p.y, p.z + gradient_delta.z) * voxel_size_inv);
    // float Fz2 = interpolate(volume, make_float3(p.x, p.y, p.z - gradient_delta.z) * voxel_size_inv);
    // n.z = __fdividef(Fz1 - Fz2, gradient_delta.z);

    return normalized(n);
}
__global__ void raycast_kernel(Patch<uint16_t> depth, Patch<float4> normals, struct Grid grid, struct Intr intr, device::Aff3f aff, device::Mat3f Rinv, float *pose)

// __global__ void integrate(Patch<uint16_t> pdepth, Patch<uchar3> rgb, uint8_t *rgbda, struct Grid grid, Intr intr, float *pose)
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

        //     //计算体素在相机坐标系的坐标
        float tmp_pt[3] = {0};
        tmp_pt[0] = vol_world.x - cam2base[0 * 4 + 3];
        tmp_pt[1] = vol_world.y - cam2base[1 * 4 + 3];
        tmp_pt[2] = vol_world.z - cam2base[2 * 4 + 3];
        float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

        // float3 vp =(vol_world-aff.t) ((aff.R * intr.cam2world(x, y, tcurr)) + aff.t);

        if (pt_cam_z <= 0)
            continue;
        int pt_pix_x = rintf(intr.fx * (pt_cam_x / pt_cam_z) + intr.cx);
        int pt_pix_y = rintf(intr.fy * (pt_cam_y / pt_cam_z) + intr.cy);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= 480)
            continue;
        depth(pt_pix_y, pt_pix_x) = pt_cam_z *10000;
    }
}
/* {
    int x = threadIdx.x;
    int y = blockIdx.x;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = 640 * 480;

    if (x >= depth.cols || y >= depth.rows)
        return;

    const float qnan = CUDART_NAN_F;

    depth(y, x) = 10000;
    normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

    float3 ray_org = aff.t;

    float time_step = 0.01f;

    for (float tcurr = 0.2; tcurr < 5.0; tcurr += time_step)
    {
        bool ret;
        float3 vp = ((aff.R * intr.cam2world(x, y, tcurr)) + aff.t) / 0.02f;
        int3 ivp;
        ivp.x = vp.x;
        ivp.y = vp.y;
        ivp.z = vp.z;
        if (ivp.x > 500 || ivp.y > 500 || ivp.z > 500)
            continue;
        if (ivp.x < 0 || ivp.y < 0 || ivp.z < 0)
            continue;
        // printf("%d %d %d\n", ivp.x, ivp.y, ivp.z);
        // struct Vovel tsdf_next = grid(ivp.x, ivp.y, ivp.z);
        // if (tsdf_next.tsdf < 0)
            depth(y, x) = tcurr * 5000;
        // if (!ret)
        //     continue;
        //   printf("%f %f %f\n", tsdf_next.tsdf,0,0);
    }
    // printf("%f %f %f\n", ray_dir.x,ray_dir.y,ray_dir.z);

    // We do subtract voxel size to minimize checks after
    // Note: origin of volume coordinate is placed
    // // in the center of voxel (0,0,0), not in the corner of the voxel!
    // float3 box_max = grid.size * grid.sca;

    // float tmin, tmax;
    // intersect(ray_org, ray_dir, box_max, tmin, tmax);

    // const float min_dist = 0.f;
    // tmin = fmaxf(min_dist, tmin);
    // if (tmin >= tmax)
    //     return;

    // tmax -= time_step;
    // float3 vstep = ray_dir * time_step;
    // float3 next = ray_org + ray_dir * tmin;
    // bool ret;
    // struct Vovel tsdf_next = grid.fetch(next, ret);
    // float voxel_size_inv = 1.0f / grid.sca;
    // for (float tcurr = tmin; tcurr < tmax; tcurr += time_step)
    // {
    //     struct Vovel tsdf_curr = tsdf_next;
    //     float3 curr = next;
    //     next += vstep;

    //     tsdf_next = grid.fetch(next, ret);
    //     if (ret == false)
    //         continue;

    //     if (tsdf_curr.tsdf < 0.f && tsdf_next.tsdf > 0.f)
    //         break;
    //     if (tsdf_curr.tsdf > 0.f && tsdf_next.tsdf < 0.f)
    //     {
    //         float Ft = grid.interpolate(curr * voxel_size_inv);
    //         float Ftdt = grid.interpolate(next * voxel_size_inv);

    //         float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);

    //         float3 vertex = ray_org + ray_dir * Ts;
    //         float3 normal = compute_normal(grid, vertex);

    //         if (!::isnan(normal.x * normal.y * normal.z))
    //         {
    //             normal = Rinv * normal;
    //             vertex = Rinv * (vertex - aff.t);

    //             normals(y, x) = make_float4(normal.x, normal.y, normal.z, 0);
    //             depth(y, x) = static_cast<ushort>(vertex.z * 1000);
    //             printf("%f\n", vertex.z);
    //         }
    //         break;
    //     }
    // }
} */
__global__ void raycast_kerne2l(Patch<uint16_t> depth, Patch<float4> normals, struct Grid grid, struct Intr intr, device::Aff3f aff, device::Mat3f Rinv)
{
    // int x = threadIdx.x;
    // int y = blockIdx.x;

    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;
    // int size = 640 * 480;

    // if (x >= depth.cols || y >= depth.rows)
    //     return;

    // float3 start_pt = aff.t;
    // for (int i = index; i < size; i += stride)
    // {
    //     float current_depth = 0;
    //     while (current_depth < 5)
    //     {
    //         float3 point =intr.cam2world(x, y, current_depth);// GetPoint3d(i, current_depth, sensor);
    //         point = camera_pose * point;
    //         Voxel v = volume->GetInterpolatedVoxel(point);
    //         if (v.weight == 0)
    //         {
    //             current_depth += volume->GetOptions().truncation_distance;
    //         }
    //         else
    //         {
    //             current_depth += v.sdf;
    //         }
    //         if (v.weight != 0 && v.sdf < volume->GetOptions().voxel_size)
    //             break;
    //     }
    //     if (current_depth < volume->GetOptions().max_sensor_depth)
    //     {
    //         float3 point = GetPoint3d(i, current_depth, sensor);
    //         point = camera_pose * point;
    //         Voxel v = volume->GetInterpolatedVoxel(point);
    //         virtual_rgb[i] = v.color;
    //     }
    //     else
    //     {
    //         virtual_rgb[i] = make_uchar3(0, 0, 0);
    //     }
    // }
}
