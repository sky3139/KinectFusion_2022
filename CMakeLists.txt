cmake_minimum_required(VERSION 3.5)

project(kinect)
# add_definitions(-w)
# ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;;-O3;-w;-gencode arch=compute_61,code=sm_61;-Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant")#;-G;-g$
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++11;-O3;-w)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	${EIGEN3_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)
cuda_add_executable(main main.cu  device.cu device.cuh 
)
target_link_libraries(main   
${OpenCV_LIBS}
${CUDA_LIBS}
) 
# cuda_add_executable(main main.cu  
# )
# target_link_libraries(main   
# ${OpenCV_LIBS}
# ${CUDA_LIBS}
# ) 
