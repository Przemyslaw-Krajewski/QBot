/*
 * NeuralNetworkArray.cpp
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#include "HashMap.h"

/*
 *
 */
HashMap::HashMap(int t_nActions, std::vector<int> t_dimensionsSize)
{
	changeIndex = t_nActions;
	maxValues = t_dimensionsSize;
	for(int i=0; i<t_nActions+1; i++) cache.push_back(CacheMemory());
}

/*
 *
 */
HashMap::~HashMap()
{

}

/*
 *
 */
double HashMap::getValue(State t_state, int t_action)
{
	double result;
	if(cache[t_action].count(t_state) > 0) result = cache[t_action].find(t_state)->second;
	else result = 2400;

	return result;
}

/*
 *
 */
std::vector<double> HashMap::getValues(State t_state)
{
	std::vector<double> result;
	for(int a=0; a<cache.size()-1;a++) result.push_back(getValue(t_state,a));

	return result;
}

/*
 *
 */
double HashMap::getChange(State t_state)
{
	double result;
	if(cache[changeIndex].count(t_state) > 0) result = cache[changeIndex].find(t_state)->second;
	else result = MAX_CHANGE;

	return result;
}

/*
 *
 */
void HashMap::setValue(State t_state, int t_action, double t_value)
{
	(cache[changeIndex])[t_state] = t_value - (cache[t_action])[t_state];
	(cache[t_action])[t_state] = t_value;
}
