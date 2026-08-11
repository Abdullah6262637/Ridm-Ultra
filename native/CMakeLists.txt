cmake_minimum_required(VERSION 3.18)
project(ridm_kernels LANGUAGES CXX)

add_library(ridm_kernels SHARED ridm_kernels.cpp)
target_compile_features(ridm_kernels PRIVATE cxx_std_17)
target_compile_definitions(ridm_kernels PRIVATE RIDM_KERNELS_BUILD)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ridm_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(ridm_kernels PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
