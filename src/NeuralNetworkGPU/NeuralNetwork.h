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

#include "Layer/NNLayer.h"

namespace NeuralNetworkGPU {


	class NeuralNetwork {
		public:
			NeuralNetwork();
			virtual ~NeuralNetwork();

			//Configuration
			void addLayer(NNLayer *t_newLayer);
			NeuronsPtr getLastLayerNeuronRef();
			TensorSize getLastLayerTensorSize();

			//basic
			std::vector<double> determineOutput(std::vector<double> x);
			std::vector<double> determineOutput(std::vector<int> x);
			std::vector<double> getOutput();
			void learnBackPropagation(std::vector<double>& z);

			//save load
			void saveToFile();
			void loadFromFile();

			//helping
		protected:
			std::vector<double> determineY();
		public:
			//debug
//			void drawNeuralNetwork() {for(auto it : layers) it->drawLayer();}

		protected:
			std::list<NNLayer*> layers;
	};

} /* namespace NeuralNetworkGPU */

#endif /* SRC_NEURALNETWORKGPU_NEURALNETWORK_H_ */
