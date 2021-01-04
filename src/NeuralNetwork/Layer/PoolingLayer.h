//
// Created by przemo on 02.01.2020.
//

#ifndef QBOT_POOLINGLAYER_H
#define QBOT_POOLINGLAYER_H

#include "NNLayer.h"
#include "../Neuron/PoolingNeuron.h"
#include "../Neuron/InputNeuron.h"

namespace NeuralNetworkCPU
{
	class PoolingLayer : public NNLayer
	{
	public:
		PoolingLayer(MatrixSize t_outputLayerSize, TensorSize t_inputSize, std::vector<Neuron*> t_prevLayerReference);
		~PoolingLayer() override = default;

	public:
		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void setDelta(std::vector<double> t_z) override;
		void learnSGD() override;

		//configuration
		std::vector<Neuron*> getNeuronPtr() override;
		TensorSize getTensorOutputSize() override {return outputSize;}
		static void configure() {/*Do nothing*/}

	protected:
		static long getIndex(int x, int y, int z, int maxX, int maxY) {return x + y*maxX + z*maxY*maxX; }

	protected:
		std::vector<PoolingNeuron> neurons;
		TensorSize outputSize;

		double learnRate;
	};
}

#endif //QBOT_POOLINGLAYER_H
