/*
 * NeuralNetworkArray.cpp
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#include "HashMapArray.h"

/*
 *
 */
HashMapArray::HashMapArray(int t_nActions, std::vector<int> t_dimensionsSize)
{
	cacheSize = 200*t_nActions;
	numberOfActions = t_nActions;
	maxValues = t_dimensionsSize;
	for(int i=0; i<numberOfActions; i++) cache.push_back(CacheMemory());
}

/*
 *
 */
HashMapArray::~HashMapArray()
{
}

/*
 *
 */
double HashMapArray::getValue(std::vector<int> t_state, int t_action)
{
	double result;
	if(cache[t_action].count (t_state) > 0)
	{
		result = cache[t_action].find(t_state)->second;
	}
	else
	{
		result = 150-30*t_action;
	}

	return result;
}

/*
 *
 */
std::vector<double> HashMapArray::getValues(std::vector<int> t_state)
{
	std::vector<double> result;
	for(int a=0; a<cache.size();a++)
	{
		result.push_back(getValue(t_state,a));
	}

	return result;
}

/*
 *
 */
void HashMapArray::setValue(std::vector<int> t_state, int t_action, double t_value)
{
	(cache[t_action])[t_state] = t_value;
}


std::vector<int> HashMapArray::getRandomState()
{

	int randomAction = rand()%numberOfActions;
	while(cache[randomAction].size()<=0) randomAction = rand()%numberOfActions;

	auto it = cache[randomAction].begin();
	int ad = rand()%(cache[randomAction].size());
	std::advance(it, ad);
	return it->first;
}
