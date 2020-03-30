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

#include "../Bot/Common.h"

using CacheMemory = std::map<std::vector<int>,double>;

class HashMap
{
public:
	HashMap(int t_nActions, std::vector<int> t_dimensionsSize);
	~HashMap();

	double getValue(State t_state, int t_action);
	std::vector<double> getValues(State t_state);
	void setValue(State t_state, int t_action, double t_value);
	double getChange(State t_state);

	long getSize() {long sum = 0; for(int i=0;i<cache.size();i++) sum+=cache[i].size(); return sum;}

	void saveToFile();
	void loadFromFile();
	std::vector<State> getStateList();

private:
	std::vector<CacheMemory> cache;
	std::vector<int> maxValues;
	int changeIndex;

	const int MAX_CHANGE = 9999;
};

#endif /* SRC_ARRAYS_HASHMAPARRAY_H_ */
