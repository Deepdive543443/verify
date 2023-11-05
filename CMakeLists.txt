cmake_minimum_required(VERSION 3.0.0)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

file(GLOB detkit_SRC
     "*.h"
     "*.cpp"
)

add_executable(detkit ${detkit_SRC})

set(ncnn_DIR /home/justinfung/cpp_libs/ncnn_x86/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(detkit ncnn)

target_include_directories(detkit PRIVATE "/home/justinfung/cpp_libs/rapidjson-1.1.0/include")



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
