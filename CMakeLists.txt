cmake_minimum_required(VERSION 3.10)
project(MLP_Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/eigen) 
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/fmt) 

set(SOURCES
    src/Activations.cpp
    src/Layer.cpp
    src/MLP.cpp
    src/Utilities.cpp
    src/Losses.cpp
    src/Optimizer.cpp
)

add_executable(train
    src/main_train.cpp
    ${SOURCES}
)

add_executable(predict
    src/main_predict.cpp
    ${SOURCES}
)
