cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# file(GLOB detkit_SRC
#      "*.h"
#      "*.cpp"
# )

file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.h )
add_library( detkit ${LIB_SOURCES} ${LIB_HEADERS} )

set(ncnn_DIR /home/justinfung/cpp_libs/ncnn_x86/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(detkit ncnn)

target_include_directories(detkit PRIVATE "/home/justinfung/cpp_libs/rapidjson-1.1.0/include")


# Build Fastest test
add_executable(Fastest_test main/fastest_test.cpp)
target_link_libraries(Fastest_test detkit)
target_include_directories(Fastest_test PRIVATE lib)
target_include_directories(Fastest_test PRIVATE "/home/justinfung/cpp_libs/rapidjson-1.1.0/include")

# Build Nanodet test
add_executable(Nano_test main/nano_test.cpp)
target_link_libraries(Nano_test detkit)
target_include_directories(Nano_test PRIVATE lib)
target_include_directories(Nano_test PRIVATE "/home/justinfung/cpp_libs/rapidjson-1.1.0/include")


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
