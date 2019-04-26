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

#include "InputNeuron.h"
#include "Neuron.h"

class NeuralNetwork {
public:
	NeuralNetwork(std::vector<double*> t_input, std::vector<int> t_layers,std::vector<double> t_n, double t_b);
	virtual ~NeuralNetwork();

	//basic
	std::vector<double> determineY();
	std::vector<double> getY();
	void learnBackPropagation(std::vector<double> z);

	//additional
	void modifyLearningRate(double v);

	//display
	void displayNeuralNetwork();
	void writeNeuronsToFile();
	void printNeuralNetworkInfo();

	double oneValue;
private:
	std::list<InputNeuron> inputLayer;
	std::list<std::vector<Neuron>> hiddenLayers;
	std::vector<double> n;
	double b;
};

#endif /* SRC_NEURALNETWORK_NEURALNETWORK_H_ */
