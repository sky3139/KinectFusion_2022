cmake_minimum_required(VERSION 3.5)

project(kinect)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-gencode arch=compute_61,code=sm_61)#;-G;-g$

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	${EIGEN3_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)
cuda_add_executable(main main.cu 
)
target_link_libraries(main   
${OpenCV_LIBS}
${CUDA_LIBS}
) 
