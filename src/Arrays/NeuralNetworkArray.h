/*
 * NeuralNetworkArray.h
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#ifndef SRC_ARRAYS_NEURALNETWORKARRAY_H_
#define SRC_ARRAYS_NEURALNETWORKARRAY_H_

#include <iostream>
#include <vector>

#include "Array.h"
#include "../NeuralNetwork/NeuralNetwork.h"

class NeuralNetworkArray : public Array
{
public:
	NeuralNetworkArray(int t_nActions, std::vector<int> t_dimensionsSize);
	virtual ~NeuralNetworkArray();

	virtual double getValue(std::vector<int> t_state, int t_action);
	virtual std::vector<double> getValues(std::vector<int> t_state);
	virtual void setValue(std::vector<int> t_state, int t_action, double t_value);

	void setInputValues(std::vector<int> t_state);

	virtual void printInfo() { /* */ }

	NeuralNetwork *neuralNetwork;
	std::vector<double> input;
	std::vector<int> maxValues;
};

#endif /* SRC_ARRAYS_NEURALNETWORKARRAY_H_ */
