/*
 * ConvolutionalNeuralNetwork.h
 *
 *  Created on: 11 lis 2019
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORK_CONVOLUTIONALNEURALNETWORK_H_
#define SRC_NEURALNETWORK_CONVOLUTIONALNEURALNETWORK_H_

#include "NeuralNetwork.h"

//#define PRINT_CONV_NEURON_CONNETIONS

struct LayerInfo
{
public:
	LayerInfo(int t_width, int t_height, int t_depth)
	{
		height=t_height;
		width=t_width;
		depth=t_depth;
	}

	int height;
	int width;
	int depth;
};

class ConvolutionalNeuralNetwork : public NeuralNetwork {
public:
	ConvolutionalNeuralNetwork(LayerInfo t_inputSize, std::vector<LayerInfo> t_layers,
							   std::vector<int> filterSize, std::vector<double> t_n, double t_b);
	virtual ~ConvolutionalNeuralNetwork();
};

#endif /* SRC_NEURALNETWORK_CONVOLUTIONALNEURALNETWORK_H_ */
