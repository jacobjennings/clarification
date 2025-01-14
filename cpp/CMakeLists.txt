cmake_minimum_required(VERSION 3.30)
project(cpp)

include(FetchContent)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "86;89;90")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

FetchContent_Declare(
        pytorch
        URL https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
)
FetchContent_MakeAvailable(pytorch)

find_package(Torch REQUIRED PATHS ${pytorch_SOURCE_DIR} NO_DEFAULT_PATH)

fetchcontent_declare(
        torchaudio
        URL https://github.com/pytorch/audio/archive/refs/tags/v2.5.0.zip
)
FetchContent_MakeAvailable(torchaudio)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

add_executable(clarification-dataloader
        mains/main.cpp
        src/ClarificationDataset.cpp
        src/ClarificationDataset.h
)
target_include_directories(clarification-dataloader PRIVATE
        ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(clarification-dataloader "${TORCHAUDIO_LIBRARY}" "${TORCH_LIBRARIES}")

add_executable(clarification-tests
        tests/ClarificationDatasetTest.cpp
        tests/ClarificationDatasetTest.h
)

target_include_directories(clarification-tests PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${Torch_DIR}/../../../../googletest-src/googletest/include
        ${torchaudio_SOURCE_DIR}
)
target_link_libraries(clarification-tests PRIVATE "${TORCH_LIBRARIES}" gtest_main)

# Enabled testing and add test tests/ClarificationDatasetTest
enable_testing()
add_test(NAME ClarificationDatasetTest COMMAND clarification-dataloader)
