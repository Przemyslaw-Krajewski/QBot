SET(CMAKE_INSTALL_PREFIX .)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

PROJECT(QBot)
cmake_minimum_required (VERSION 2.8)

find_package( OpenCV REQUIRED )
find_package( OpenMP )

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_executable(QBot 
			src/main.cpp 
			src/Game/Game.cpp 
			src/QLearning/QLearning.cpp
			src/ActionCritic/ActionCritic.cpp
			src/Arrays/Array.cpp
			src/Arrays/Table.cpp
			src/Arrays/NeuralNetworkArray.cpp
			src/Arrays/HybridArray.cpp
			src/Arrays/HashMapArray.cpp
			src/NeuralNetwork/InputNeuron.cpp
			src/NeuralNetwork/Neuron.cpp
			src/NeuralNetwork/NeuralNetwork.cpp )

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( QBot ${OpenCV_LIBRARIES} pthread)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Grafika DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
