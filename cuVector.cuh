#pragma once
#include <cuda.h>
#include <cassert>
#include <iostream>
#include "utils/cutil_math.h"
// struct Point
// {
//     T x, y, z;
//     Point() : x(0), y(0), z(0) {}
//     Point(T x, T y, T z) : x(x), y(y), z(z) {}
//     T crass(const Point p, const Point q, const Point r)
//     {
//         return (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x);
//     }
// };
// typedef Point<float> Pointf;
// typedef Point<int> Pointi;

template <class T>
struct cuBase
{
    T *devPtr;
    uint64_t capacity;
    uint64_t size;
    //拷贝构造函数
    cuBase(const cuBase &lth) : devPtr(lth.devPtr), capacity(lth.capacity), size(lth.size){};
    // struct cuBase copy()
    // {
    //     cuBase b;
    //     //   :  devPtr(lth.devPtr), capacity(lth.capacity), size(lth.size)
    //     return cuBase;
    // }
};
template <class T>
struct cuVector
{
public:
    cuBase<T> *cb;
    __host__ cuVector(int capacity)
    {

        ck(cudaMallocManaged((void **)&cb, sizeof(cuBase<T>)));
        cb->capacity = capacity;
        cb->size = 0;
        ck(cudaMallocManaged((void **)&(cb->devPtr), sizeof(T) * cb->capacity));
        ck(cudaMemset(cb->devPtr, 0, sizeof(T) * cb->capacity));
    }
    __host__ cuVector(int capacity, T val)
    {
        ck(cudaMallocManaged((void **)&cb, sizeof(cuBase<T>)));
        cb->capacity = capacity;
        cb->size = capacity;
        ck(cudaMallocManaged((void **)&(cb->devPtr), sizeof(T) * cb->capacity));
        ck(cudaMemset(cb->devPtr, val, sizeof(T) * cb->capacity));
    }
    //拷贝构造函数
    __host__ __device__ cuVector(const cuVector &lth)
    {
        // cb->devPtr = lth.cb->devPtr;
        // cb->size = lth.cb->size;
        // cb->capacity = lth.cb->capacity;
        cb = lth.cb;
    }
    __host__ void release()
    {
        cudaFree(cb);
    }
    ~cuVector()
    {
        cudaFree(cb->devPtr);
        // cudaFree(cb);
    }
    __host__ __device__ void push_back(T val)
    { //多线程时有问题，需要使用原子数
        assert(cb->size < cb->capacity);
        cb->devPtr[cb->size++] = val;
    }
    __host__ __device__ inline T &operator[](size_t x)
    {
        return *(cb->devPtr + x);
    }
    __host__ __device__ inline size_t size()
    {
        return cb->size;
    }
    __host__ __device__ inline size_t capacity()
    {
        return cb->capacity;
    }
    __host__ __device__ inline T &back()
    {
        return *(cb->devPtr + cb->size - 1);
    }
    __host__ __device__ inline void pop_back()
    {
        assert(cb->size != 0);
        cb->size--;
    }
    void print()
    {
        std::cout << size() << " " << capacity() << std::endl;
    }
    T *begin() { return cb->devPtr; };
    T *end() { return cb->devPtr + cb->capacity; };
};
template <class T>
class cuVector2D : public cuVector<T>
{
public:
    size_t rows, cols;
    cuVector2D(int rows, int cols, T val) : cuVector<T>(rows * cols, val), rows(rows), cols(cols)
    {
        // ck(cudaMallocManaged((void **)&devPtr, sizeof(T) * size));
        // ck(cudaMemset(devPtr, 0, sizeof(T) * size));
    }
    cuVector2D(int size)
    {
        //
    }
    __device__ inline T *operator()(size_t row, size_t col)
    {
        // if (rows < pitch)
        //     return mat[x];
        return ((cuVector<T>::cb->devPtr + cols * row + col));
    }
    __host__ __device__ inline T *operator[](size_t row)
    {
        // if (rows < pitch)
        //     return mat[x];
        return cuVector<T>::cb->devPtr + cols * row;
    }
};
template <class T>
struct Patch
{
    T *devPtr;
    size_t pitch = 0;
    size_t rows, cols;
    Patch()
    {
    }
    Patch(int rows, int cols) : rows(rows), cols(cols)
    {
        ck(cudaMallocPitch((void **)&devPtr, &pitch, cols * sizeof(T), rows));
        ck(cudaMemset2D(devPtr, pitch, 0, sizeof(T) * cols, rows));
        // std::cout << pitch << " " << sizeof(T) * cols << std::endl;
    }
    void creat(const int rows, const int cols)
    {
        this->rows = rows;
        this->cols = cols;
        ck(cudaMallocPitch((void **)&devPtr, &pitch, cols * sizeof(T), rows));
        ck(cudaMemset2D(devPtr, pitch, 0, sizeof(T) * cols, rows));
    }
    //拷贝构造函数
    __host__ __device__ Patch(const Patch &lth)
    {
        this->devPtr = lth.devPtr;
        this->rows = lth.rows;
        this->pitch = lth.pitch;
        this->cols = lth.cols;
    }

    ~Patch()
    {
        // cudaFree(devPtr);
    }
    void release()
    {
        cudaFree(devPtr);
    }
    __host__ void upload(const T *host_ptr_arg, size_t host_step_arg)
    {
        ck(cudaMemcpy2D(devPtr, pitch, host_ptr_arg, host_step_arg, cols * sizeof(T), rows, cudaMemcpyHostToDevice));
    }
    __host__ void download(const T *host_ptr_arg, size_t host_step_arg)
    {
        // if (host_step_arg == 0 || devPtr == nullptr || pitch == 0)
        {
            printf("%x,%d %x,%ld\n", host_ptr_arg, host_step_arg, devPtr, pitch);
        }
        ck(cudaMemcpy2D((void *)host_ptr_arg, host_step_arg, devPtr, pitch, sizeof(T) * cols, rows, cudaMemcpyDeviceToHost));
    }
    __device__ inline T &operator()(size_t rows, size_t cols)
    {
        return devPtr[rows * pitch / sizeof(T) + cols];
    }
    __device__ inline  T *get(size_t rows, size_t cols)
    {
        // if (rows < pitch)
        //     return mat[x];
        return &devPtr[rows * pitch /sizeof(T)+ cols];
    }

    __host__ void print()
    {
        printf("pitch=%ld rows=%ld cols=%ld\n", pitch, rows, cols);
    }
};