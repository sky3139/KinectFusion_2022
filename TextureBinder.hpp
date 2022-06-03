#pragma once
#include <iostream>
#include "cuVector.cuh"

namespace Cuda
{
    class TextureBinder
    {
    public:
        cudaTextureObject_t obj;
        struct cudaResourceDesc resDesc;
        struct cudaTextureDesc texDesc;
        template <class T>
        void init(const Patch<T> &arr)
        {
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypePitch2D;
            resDesc.res.pitch2D.pitchInBytes = arr.pitch;
            resDesc.res.pitch2D.devPtr = arr.devPtr;
            resDesc.res.pitch2D.width = arr.cols;
            resDesc.res.pitch2D.height = arr.rows;
            resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>(); // cudaCreateChannelDescHalf(); u20
            // resDesc.res.pitch2D.desc = cudaCreateChannelDescHalf(); // cudaCreateChannelDesc<T>(); //  u16
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeBorder;
            texDesc.addressMode[1] = cudaAddressModeBorder;
            texDesc.addressMode[2] = cudaAddressModeBorder;
            texDesc.filterMode =cudaFilterModePoint ; // cudaFilterModePoint;cudaFilterModeLinear
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;
            cudaCreateTextureObject(&obj, &resDesc, &texDesc, NULL);
            ck(cudaGetLastError());
        }
        template <class T>
        TextureBinder(const T &arr)
        {
            init(arr);
        }
        ~TextureBinder()
        {
            cudaDestroyTextureObject(obj);
        }
    };
}
