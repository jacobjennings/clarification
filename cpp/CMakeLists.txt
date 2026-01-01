cmake_minimum_required(VERSION 3.30)
project(cpp)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "86;89;90")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output" FORCE)

# Find OpenMP
find_package(OpenMP REQUIRED)

find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBOPUS REQUIRED opus)
pkg_check_modules(LIBLZ4 REQUIRED liblz4)  # Add this line to find liblz4



#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

FetchContent_Declare(
        pytorch
        URL https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
        URL_HASH SHA256=08552b1d13de1389d01c64aaa5e2c6c5092f922618b0d32ccff8ed9099d75735
)
FetchContent_MakeAvailable(pytorch)

find_package(Torch REQUIRED PATHS ${pytorch_SOURCE_DIR} NO_DEFAULT_PATH)

fetchcontent_declare(
        torchaudio
        URL https://github.com/pytorch/audio/archive/refs/tags/v2.5.0.zip
)
FetchContent_MakeAvailable(torchaudio)

add_executable(clarification-dataloader
        mains/main.cpp
        src/ClarificationDataset.cpp
        src/ClarificationDataset.h
        src/ClarificationLz4Dataset.cpp
        src/ClarificationLz4Dataset.h
        src/ClarificationOpusDataset.cpp
        src/ClarificationOpusDataset.h
)
target_include_directories(clarification-dataloader PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${LIBLZ4_INCLUDE_DIRS}
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Download and build FFmpeg ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Download FFmpeg
FetchContent_Declare(
        ffmpeg
        #        GIT_REPOSITORY https://git.ffmpeg.org/ffmpeg.git
        #        GIT_TAG release/6.1
        GIT_REPOSITORY https://github.com/jacobjennings/FFmpeg.git
        GIT_TAG 13a9494c108beb08cb88fd2161c0de93a8dcb179
)

FetchContent_MakeAvailable(ffmpeg)

# Set FFmpeg build and install directories
set(FFMPEG_BUILD_DIR ${ffmpeg_SOURCE_DIR}/build)
set(FFMPEG_INSTALL_DIR ${CMAKE_BINARY_DIR}/ffmpeg_install)

# Create the build directory
file(MAKE_DIRECTORY ${FFMPEG_BUILD_DIR})

# Configure and build FFmpeg using its configure script
    execute_process(
        COMMAND
        ./configure --prefix=${FFMPEG_INSTALL_DIR}
        --disable-static
        --enable-shared
        --disable-vaapi
        --enable-libopus
        --extra-cflags="-I/usr/local/include"
        --extra-ldflags="-L/usr/local/lib"
        WORKING_DIRECTORY ${ffmpeg_SOURCE_DIR}
        RESULT_VARIABLE FFMPEG_CONFIGURE_RESULT
)
if (FFMPEG_CONFIGURE_RESULT)
    message(FATAL_ERROR "FFmpeg configure failed with result ${FFMPEG_CONFIGURE_RESULT}")
endif()

execute_process(
        COMMAND make -j${CMAKE_NUMBER_OF_PROCESSORS}
        WORKING_DIRECTORY ${ffmpeg_SOURCE_DIR}
        RESULT_VARIABLE FFMPEG_MAKE_RESULT
)
if (FFMPEG_MAKE_RESULT)
    message(FATAL_ERROR "FFmpeg make failed with result ${FFMPEG_MAKE_RESULT}")
endif()

execute_process(
        COMMAND make install
        WORKING_DIRECTORY ${ffmpeg_SOURCE_DIR}
        RESULT_VARIABLE FFMPEG_INSTALL_RESULT
)
if (FFMPEG_INSTALL_RESULT)
    message(FATAL_ERROR "FFmpeg install failed with result ${FFMPEG_INSTALL_RESULT}")
endif()

# Add FFmpeg include directories and libraries to your target
target_include_directories(clarification-dataloader PRIVATE
        ${FFMPEG_INSTALL_DIR}/include
)

set(FFMPEG_LIBRARIES
        ${LIBOPUS_LIBRARIES}
        ZLIB::ZLIB
        -L${FFMPEG_INSTALL_DIR}/lib
        -lavcodec
        -lavformat
        -lavutil
        -lswresample
        -lswscale
)
set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INSTALL_DIR}/include)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG HEAD
)
FetchContent_MakeAvailable(googletest)

# Link OpenMP
target_link_libraries(clarification-dataloader
        -pthread -lvdpau -lX11 -lm -latomic -lX11 -lbz2 -lz
        ${FFMPEG_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${TORCHAUDIO_LIBRARY}
        ${X11_LIBRARIES}
        ${LIBLZ4_LIBRARIES}
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX> # Conditionally link OpenMP
)

add_executable(clarification-tests
        tests/ClarificationDatasetTest.cpp
        tests/ClarificationDatasetTest.h
)

target_include_directories(clarification-tests PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${googletest_SOURCE_DIR}/googletest/include
        ${torchaudio_SOURCE_DIR}
        ${FFMPEG_INCLUDE_DIRS}
        ${LIBLZ4_INCLUDE_DIRS}
)

target_link_libraries(clarification-tests PRIVATE
        -pthread -lvdpau -lX11 -lm -latomic -lX11 -lbz2 -lz
        ${FFMPEG_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${TORCHAUDIO_LIBRARY}
        ${X11_LIBRARIES}
        ${LIBLZ4_LIBRARIES}
        gtest
        gtest_main
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX> # Conditionally link OpenMP
)


# Enabled testing and add test tests/ClarificationDatasetTest
enable_testing()
add_test(NAME ClarificationDatasetTest COMMAND clarification-dataloader)