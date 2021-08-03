//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_POOLINGLAYER_H
#define NEURALNETWORKGPU_POOLINGLAYER_H

#include "NNLayer.h"

namespace NeuralNetworkGPU
{
	class PoolingLayer : public NNLayer
	{
	public:
		PoolingLayer(NeuronsPtr t_prevLayerReference);
		virtual ~PoolingLayer();

	public:
		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void learnSGD() override;
		void learnAdam() override;

		//configuration
		NeuronsPtr getNeuronPtr() override;

		//save load
//		void saveToFile(std::ofstream &t_file) override;
//		void loadFromFile(std::ifstream &t_file) override;

	protected:
		float *de_input;
		TensorSize *d_inputSize, inputSize;

		float *d_output,*output;
		TensorSize size;

		float *d_deltas, *deltas;
		float *de_prevDeltas;
	};
}

#endif //NEURALNETWORKGPU_POOLINGLAYER_H
