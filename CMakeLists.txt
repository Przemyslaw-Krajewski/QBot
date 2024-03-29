cmake_minimum_required (VERSION 2.8)
PROJECT(QBot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
find_package( X11 REQUIRED )
find_package(OpenMP)
find_package(CUDA)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(QBot
		    src/main.cpp
			src/Bot/Bot.cpp
			src/Bot/State.cpp
			src/ReinforcementLearning/ReinforcementLearning.cpp
			src/ReinforcementLearning/QLearning.cpp
			src/ReinforcementLearning/GeneralizedQL.cpp
			src/ReinforcementLearning/ActorCriticNN.cpp
			src/Analyzers/StateAnalyzer/StateAnalyzer.cpp
			src/Analyzers/ImageAnalyzer/ImageAnalyzer.cpp
			src/Analyzers/ImageAnalyzer/RawImageAnalyzer.cpp
			src/Analyzers/ImageAnalyzer/MetaDataAnalyzer.cpp
			src/Analyzers/MemoryAnalyzer.cpp
			src/HashMap/HashMap.cpp
			src/Loggers/DataDrawer.cpp
			src/Loggers/StateViewer.cpp
			src/NeuralNetwork/NeuralNetwork.cpp
			src/NeuralNetwork/Neuron/AdaptiveNeuron.cpp
			src/NeuralNetwork/Neuron/InputNeuron.cpp
			src/NeuralNetwork/Neuron/PoolingNeuron.cpp
			src/NeuralNetwork/Layer/InputLayer.cpp
			src/NeuralNetwork/Layer/SigmoidLayer.cpp
			src/NeuralNetwork/Layer/ConvolutionalLayer.cpp
			src/NeuralNetwork/Layer/ModifiedConvolutionalLayer.cpp
			src/NeuralNetwork/Layer/PoolingLayer.cpp
			src/NeuralNetworkGPU/NeuralNetwork.cpp
			src/NeuralNetworkGPU/Layer/InputLayer.cu
			src/NeuralNetworkGPU/Layer/SigmoidLayer.cu
			src/NeuralNetworkGPU/Layer/ConvolutionalLayer.cu
			src/NeuralNetworkGPU/Layer/ConvSeparateWeightsLayer.cu
			src/NeuralNetworkGPU/Layer/PoolingLayer.cu
			src/NeuralNetworkGPU/Layer/FuseLayer.cu)
			
include_directories(${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries( QBot ${OpenCV_LIBRARIES} ${X11_LIBRARIES} pthread)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graphics DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
