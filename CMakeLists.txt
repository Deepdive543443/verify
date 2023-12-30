cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.h )
add_library( detkit ${LIB_SOURCES} ${LIB_HEADERS} )

set(ncnn_DIR ../../cpp_libs/ncnn-20231027-full-source/build/install/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(detkit ncnn)

set(OpenCV_DIR ../../cpp_libs/opencv-mobile-2.4.13.7-ubuntu-2204/share/OpenCV CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(OpenCV REQUIRED core imgproc)


set(RPJSON_DIR "../../cpp_libs/rapidjson-1.1.0/include")

target_include_directories(detkit PRIVATE ${RPJSON_DIR})


# Build Fastest test
add_executable(Fastest_test main/fastest_test.cpp)
target_link_libraries(Fastest_test detkit)
target_include_directories(Fastest_test PRIVATE lib)
target_include_directories(Fastest_test PRIVATE ${RPJSON_DIR})

# Build Nanodet test
add_executable(Nano_test main/nano_test.cpp)
target_link_libraries(Nano_test detkit)
target_include_directories(Nano_test PRIVATE lib)
target_include_directories(Nano_test PRIVATE ${RPJSON_DIR})

# Build shape test
add_executable(Shape_test main/shape_test.cpp)
target_link_libraries(Shape_test detkit)
target_include_directories(Shape_test PRIVATE lib)
target_include_directories(Shape_test PRIVATE ${RPJSON_DIR})

# Build new inference test
add_executable(nanop_intra main/nanop_intra.cpp)
target_link_libraries(nanop_intra detkit)
target_include_directories(nanop_intra PRIVATE lib)
target_include_directories(nanop_intra PRIVATE ${RPJSON_DIR})
target_link_libraries(nanop_intra opencv_core opencv_highgui)

# set(OpenCV_DIR ../../cpp_libs/opencv-mobile-2.4.13.7-ubuntu-2204/share/OpenCV CACHE PATH "Directory that contains ncnnConfig.cmake")
# find_package(OpenCV REQUIRED core imgproc)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
