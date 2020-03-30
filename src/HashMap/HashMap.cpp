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
	else result = 200-20*t_action;

	return result;
}

/*
 *
 */
std::vector<double> HashMap::getValues(State t_state)
{
	std::vector<double> result;
	for(int a=0; a<cache.size();a++)
	{
		if(a==changeIndex) continue;
		result.push_back(getValue(t_state,a));
	}

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

/*
 *
 */
std::vector<State> HashMap::getStateList()
{
	std::vector<State> result;
	for(int i=0; i<cache.size(); i++)
	{
		if(i == changeIndex) continue;
		for(auto map_it = cache[i].begin(); map_it != cache[i].end(); ++map_it)
		{
			result.push_back(map_it->first);
		}
	}
	return result;
}

/*
 *
 */
void HashMap::saveToFile()
{
	std::remove("QValues.dat");
	std::ofstream file("QValues.dat");

	double numberOfActions = cache.size();
	double stateSize = maxValues.size();

	file << numberOfActions << " ";
	file << stateSize << " ";
	file << (double) changeIndex << " ";

	for(int i=0; i<cache.size(); i++)
	{
		for(auto map_it = cache[i].begin(); map_it != cache[i].end(); ++map_it) {
			file << i << " ";
			file << map_it->second << " ";

			if(map_it->first.size() != stateSize) std::cout << "STAN ZAPISU MAPY: UPSIK!!!";
			for(int j=0; j<map_it->first.size(); j++)
			{
				file << (map_it->first)[j] << " ";
			}
		}
	}

	file.close();
}

/*
 *
 */
void HashMap::loadFromFile()
{
	std::ifstream file("QValues.dat");

	double numberOfActions;
	double stateSize;
	double doubleBuff;

	file >> numberOfActions;
	file >> stateSize;
	file >> doubleBuff; changeIndex = doubleBuff;

	maxValues = std::vector<int>(stateSize,1);
	cache.clear();
	for(int i=0; i<numberOfActions; i++) cache.push_back(CacheMemory());

	while(file >> doubleBuff)
	{
		int cacheIndex = doubleBuff;
		if(file.eof()) {std::cout << "KONIEC PLIKU\n"; return;}
		double value; file >> value;
		std::vector<int> state;
		for(int i=0; i<stateSize; i++)
		{
			if(file.eof()) {std::cout << "KONIEC PLIKU\n"; return;}
			file >> doubleBuff;
			state.push_back(doubleBuff);
		}
		(cache[cacheIndex])[state] = value;
	}

	file.close();
}
