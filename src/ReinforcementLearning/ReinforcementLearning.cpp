/*
 * ReinforcementLearning.cpp
 *
 *  Created on: 16 wrz 2020
 *      Author: przemo
 */

#include "ReinforcementLearning.h"

ReinforcementLearning::ReinforcementLearning()
{
	randomNumberGenerator =std::mt19937(randomDevice());
}

ReinforcementLearning::~ReinforcementLearning()
{

}

/*
 *
 */
int ReinforcementLearning::getWeightedRandom(std::vector<double> t_array)
{
	//Sum
	double sum = 0;
	for(int i=0; i<t_array.size(); i++)
	{
		sum += t_array[i];
	}

	//Choose random
	if(sum == 0) return std::uniform_int_distribution<>(0,t_array.size())(randomNumberGenerator);
	double randomValue = std::uniform_real_distribution<>(0.0, 1.0)(randomNumberGenerator);
	for(int i=0; i<t_array.size(); i++)
	{
		randomValue -= t_array[i]/sum;
		if(randomValue < 0)
		{
			return i;
		}
	}
	return t_array.size()-1;
}
