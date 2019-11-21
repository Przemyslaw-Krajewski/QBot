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

using NNInput = std::vector<double>;

class NeuralNetwork {
protected:
	NeuralNetwork() {b=0;}
public:
	NeuralNetwork(int t_inputSize, std::vector<int> t_layers,std::vector<double> t_n, double t_b);
	NeuralNetwork(const NeuralNetwork& t_neuralNetwork);
	virtual ~NeuralNetwork();

	//basic
	std::vector<double> determineY(std::vector<double> &x);
	std::vector<double> determineY(const std::vector<int> &x);
	std::vector<double> getY();
	void learnBackPropagation(std::vector<double>& z);
	void learnBackPropagation(std::vector<double>& x, std::vector<double>& z);

	//helping
protected:
	std::vector<double> determineY();

	//getInfo
protected:
	int getInputSize() const {return inputLayer.size()-1;}
	double getActivationFunctionParameter() const {return b;}
	std::vector<double> getLearningRates() const {return n;}
	std::vector<int> getLayersLayout() const;
	const std::list<std::vector<Neuron>>* getHiddenLayers() const { return &hiddenLayers;}

public:
	//display
	void displayNeuralNetwork();
	void writeNeuronsToFile();
	void printNeuralNetworkInfo();

protected:
	std::vector<InputNeuron> inputLayer;
	std::list<std::vector<Neuron>> hiddenLayers;
	std::vector<double> n;
	double b;
};

#endif /* SRC_NEURALNETWORK_NEURALNETWORK_H_ */
