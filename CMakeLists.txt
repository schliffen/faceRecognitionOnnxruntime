cmake_minimum_required(VERSION 3.17)
project(tinyfr_onnxruntime)

set (CMAKE_CXX_STANDARD 14)




set(onnxruntime_INSTALL_PREFIX /home/ai/Projlab/onnxruntime)
set(onnxruntime_INCLUDE_DIRS
        ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
        ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
        )

#find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /home/ai/Projlab/onnxruntime/build/Linux/RelWithDebInfo/ )
find_package(OpenMP QUIET)

include_directories(${onnxruntime_INCLUDE_DIRS})
#include_directories(/home/ai/librealsense/include)

find_package( OpenCV REQUIRED )


#find_package(realsense2 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library( detector
        SHARED
        include/faceProcess.cpp
        include/faceProcess.h
        include/preprocessImg.h
)


link_directories(/home/ai/Projlab/onnxruntime/build/Linux/RelWithDebInfo/ )

add_executable(${PROJECT_NAME} main.cpp )


target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries( ${PROJECT_NAME} onnxruntime  )
target_link_libraries( ${PROJECT_NAME} detector )
