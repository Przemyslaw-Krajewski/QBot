/*
 * NeuralNetworkArray.h
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#ifndef SRC_ARRAYS_HASHMAPARRAY_H_
#define SRC_ARRAYS_HASHMAPARRAY_H_

#include <iostream>
#include <vector>
#include <map>

#include "Array.h"
#include "../NeuralNetwork/NeuralNetwork.h"

using CacheMemory = std::map<std::vector<int>,double>;

class HashMapArray : public Array
{
public:
	HashMapArray(int t_nActions, std::vector<int> t_dimensionsSize);
	virtual ~HashMapArray();

	virtual double getValue(std::vector<int> t_state, int t_action);
	virtual std::vector<double> getValues(std::vector<int> t_state);
	virtual void setValue(std::vector<int> t_state, int t_action, double t_value);

	virtual void printInfo()
	{
		for(int i=0; i<numberOfActions; i++) std::cout << cache[i].size() << "  ";
		std::cout << "\n";
	}

	std::vector<CacheMemory> cache;

	int numberOfActions;
	std::vector<int> maxValues;

	int cacheSize;
};

#endif /* SRC_ARRAYS_HASHMAPARRAY_H_ */
