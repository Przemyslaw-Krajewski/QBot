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

#include "../Bot/State.h"

using CacheMemory = std::map<std::vector<int>,double>;

class HashMap
{
public:
	HashMap(int t_nActions);
	~HashMap();

	double getValue(State t_state, int t_action);
	double getChange(State t_state);
	std::vector<double> getValues(State t_state);

	void setValue(State t_state, int t_action, double t_value);

	long getSize() {long sum = 0; for(int i=0;i<cache.size();i++) sum+=cache[i].size(); return sum;}

private:
	std::vector<CacheMemory> cache;
	int changeIndex;

	const int MAX_CHANGE = 9999;
};

#endif /* SRC_ARRAYS_HASHMAPARRAY_H_ */
