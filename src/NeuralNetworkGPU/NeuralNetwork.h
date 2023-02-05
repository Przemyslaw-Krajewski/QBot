/*
 * NeuralNetwork.h
 *
 *  Created on: 9 maj 2020
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORKGPU_NEURALNETWORK_H_
#define SRC_NEURALNETWORKGPU_NEURALNETWORK_H_

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <stdio.h>
#include <list>
#include <vector>

#include <time.h>

#include "Layer/InputLayer.h"
#include "Layer/SigmoidLayer.h"
#include "Layer/ConvolutionalLayer.h"
#include "Layer/ConvSeparateWeightsLayer.h"
#include "Layer/PoolingLayer.h"
#include "Layer/FuseLayer.h"

#include "Layer/NNLayer.h"

namespace NeuralNetworkGPU {

	using NNInput = std::vector<double>;

	enum class LearnMode {SGD, Adam};

	class NeuralNetwork {
		public:
			NeuralNetwork(LearnMode t_lm = LearnMode::SGD);
			virtual ~NeuralNetwork();

			//Configuration
			void addLayer(NNLayer *t_newLayer);
			NeuronsPtr getLastLayerNeuronRef();
			TensorSize getLastLayerTensorSize();

			//basic
			std::vector<double> determineOutput(std::vector<double> x);
			std::vector<double> determineOutput(std::vector<int> x);
			std::vector<double> getOutput();
			void learnBackPropagation();
			void setSoftMaxDelta(std::vector<double> &z, double diff, int chosen);
			void setMeanSquareDelta(std::vector<double> &z);

			//save load
			void saveToFile(std::string t_name);
			void loadFromFile(std::string t_name);

			//helping
		protected:
			std::vector<double> determineY();
		public:
			//debug
			void drawLayer(int layer);
			void printInfo() {std::cout << "NeuralNetwork:\n";for(auto it : layers) it->printInfo();}
//			void drawNeuralNetwork() {for(auto it : layers) it->drawLayer();}

		protected:
			std::list<NNLayer*> layers;
			LearnMode learnMode;
	};

} /* namespace NeuralNetworkGPU */

#endif /* SRC_NEURALNETWORKGPU_NEURALNETWORK_H_ */
