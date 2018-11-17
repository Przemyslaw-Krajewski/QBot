/*
 * NeuralNetworkArray.h
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#ifndef SRC_ARRAYS_HYBRIDARRAY_H_
#define SRC_ARRAYS_HYBRIDARRAY_H_

#include <iostream>
#include <vector>
#include <map>

#include "Array.h"
#include "../NeuralNetwork/NeuralNetwork.h"

using CacheMemory = std::map<std::vector<int>,double>;

class HybridArray : public Array
{
public:
	HybridArray(int t_nActions, std::vector<int> t_dimensionsSize);
	virtual ~HybridArray();

	virtual double getValue(std::vector<int> t_state, int t_action);
	virtual std::vector<double> getValues(std::vector<int> t_state);
	virtual void setValue(std::vector<int> t_state, int t_action, double t_value);

	NeuralNetwork* createNeuralNetwork();
	void setInputValues(std::vector<int> t_state);
	void rewriteData();

	virtual void printInfo() { /* */ }

	NeuralNetwork *neuralNetwork;
	std::vector<CacheMemory> cache;

	int numberOfActions;
	std::vector<double> input;
	std::vector<int> maxValues;

	double acceptableError;
	int cacheSize;
};

#endif /* SRC_ARRAYS_HYBRIDARRAY_H_ */
