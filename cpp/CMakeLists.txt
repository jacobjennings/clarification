cmake_minimum_required(VERSION 3.30)
project(cpp)

include(FetchContent)

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "86;89;90")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Enable verbose output" FORCE)

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
)
target_include_directories(clarification-dataloader PRIVATE
        ${TORCH_INCLUDE_DIRS}
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
        --enable-static
        --disable-shared
        --enable-libopus
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

find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBOPUS REQUIRED opus)

set(FFMPEG_LIBRARIES
        ${LIBOPUS_LIBRARIES}
        ZLIB::ZLIB
        ${FFMPEG_INSTALL_DIR}/lib/libavcodec.a
        ${FFMPEG_INSTALL_DIR}/lib/libavformat.a
        ${FFMPEG_INSTALL_DIR}/lib/libavutil.a
        ${FFMPEG_INSTALL_DIR}/lib/libswresample.a
        ${FFMPEG_INSTALL_DIR}/lib/libswscale.a
)
set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INSTALL_DIR}/include)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG HEAD
)
FetchContent_MakeAvailable(googletest)


target_link_libraries(clarification-dataloader
        ${FFMPEG_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${TORCHAUDIO_LIBRARY}
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
)

target_link_libraries(clarification-tests PRIVATE
        ${FFMPEG_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${TORCHAUDIO_LIBRARY}
        gtest
        gtest_main
)


# Enabled testing and add test tests/ClarificationDatasetTest
enable_testing()
add_test(NAME ClarificationDatasetTest COMMAND clarification-dataloader)
