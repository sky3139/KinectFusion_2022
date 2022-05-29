#pragma once

#include "utils/utils.cuh"
#include "./TSDF.cuh"
#include <math_constants.h>
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

__global__ void raycast_kernel(Patch<uint16_t> depth, Patch<float4> normals, struct Grid grid, struct Intr intr, Aff3f aff);
// {
//     raycaster(depth, normals);
// };
