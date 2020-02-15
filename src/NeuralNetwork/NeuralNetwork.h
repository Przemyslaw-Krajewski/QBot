/*
 * SiecNeuronowa.h
 *
 *  Created on: 20 gru 2017
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORK_NEURALNETWORK_H_
#define SRC_NEURALNETWORK_NEURALNETWORK_H_

//#define PRINT_NEURON_CONNETIONS

#include <vector>
#include <list>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "Layer/NNLayer.h"
#include "Layer/InputLayer.h"
#include "Layer/SigmoidLayer.h"
#include "Layer/ConvolutionalLayer.h"
#include "Layer/PoolingLayer.h"


using NNInput = std::vector<double>;

class NeuralNetwork
{
public:
    NeuralNetwork();
	virtual ~NeuralNetwork();

	//Configuration
	void addLayer(NNLayer *t_newLayer);
	std::vector<Neuron*> getLastLayerNeuronRef();
	TensorSize getLastLayerTensorSize();

	//basic
	std::vector<double> determineOutput(std::vector<double> &x);
	std::vector<double> determineOutput(const std::vector<int> &x);
	std::vector<double> getOutput();
	void learnBackPropagation(std::vector<double>& z);

	//helping
protected:
	std::vector<double> determineY();
public:
	//debug
	void drawNeuralNetwork() {for(auto it : layers) it->drawLayer();}

protected:
	std::list<NNLayer*> layers;
};

#endif /* SRC_NEURALNETWORK_NEURALNETWORK_H_ */
