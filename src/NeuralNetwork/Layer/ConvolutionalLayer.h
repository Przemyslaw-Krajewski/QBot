//
// Created by przemo on 02.01.2020.
//

#ifndef QBOT_CONVOLUTIONALLAYER_H
#define QBOT_CONVOLUTIONALLAYER_H

#include "NNLayer.h"
#include "../Neuron/AdaptiveNeuron.h"
#include "../Neuron/InputNeuron.h"

namespace CPUNeuralNetwork
{
	class ConvolutionalLayer : public NNLayer
	{

	public:
		ConvolutionalLayer(double t_learnRate, MatrixSize t_filterSize, int t_numberOfLayers, TensorSize t_inputSize,
						   std::vector<Neuron*> t_prevLayerReference);
		~ConvolutionalLayer() override = default;

	public:
		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void setDelta(std::vector<double> t_z) override;
		void learnBackPropagation() override;

		//configuration
		std::vector<Neuron*> getNeuronPtr() override;
		TensorSize getTensorOutputSize() override {return outputSize;}
		static void configure() {/*Do nothing*/}

		//visualization
		void drawLayer() override;

	protected:
		static long getIndex(int x, int y, int z, int maxX, int maxY) {return x + y*maxX + z*maxY*maxX; }

	protected:
		std::vector<AdaptiveNeuron> neurons;
		InputNeuron emptyValue;
		std::vector<std::vector<double>> filters;
		TensorSize filterSize;
		TensorSize outputSize;

		double learnRate;
	};
}


#endif //QBOT_CONVOLUTIONALLAYER_H
