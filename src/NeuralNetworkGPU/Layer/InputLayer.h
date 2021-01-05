//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_INPUTLAYER_H
#define NEURALNETWORKGPU_INPUTLAYER_H

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "NNLayer.h"

namespace NeuralNetworkGPU
{
	class InputLayer : public NNLayer
	{
	public:
		InputLayer(int t_size);
		virtual ~InputLayer();

	public:
		//input
		void setInput(std::vector<int> t_input) override;
		void setInput(std::vector<double> t_input) override;

		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void learnSGD() override ;
		void learnAdam() override ;

		//configuration
		NeuronsPtr getNeuronPtr() override;

		//save load
//		void saveToFile(std::ofstream &t_file) override;

	protected:
		float *d_input;
		float *input;
		int size;
	};
}


#endif //NEURALNETWORKGPU_INPUTLAYER_H
