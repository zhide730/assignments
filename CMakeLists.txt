cmake_minimum_required(VERSION 2.8)
project(assignments)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(useEigen SHARED src/useEigen.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(test1 examples/test1.cpp)
target_link_libraries(test1 useEigen ${OpenCV_LIBS})

add_executable(test2 examples/test2.cpp)
target_link_libraries(test2 useEigen ${OpenCV_LIBS})

add_executable(useOpencv examples/useOpencv.cpp)
target_link_libraries(useOpencv ${OpenCV_LIBS})

add_executable(test3 examples/test3.cpp)
target_link_libraries(test3 ${OpenCV_LIBS})