/*
 * ReinforcementLearning.cpp
 *
 *  Created on: 16 wrz 2020
 *      Author: przemo
 */

#include "ReinforcementLearning.h"

ReinforcementLearning::ReinforcementLearning()
{

}

ReinforcementLearning::~ReinforcementLearning()
{

}

/*
 *
 */
int ReinforcementLearning::getIndexOfMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return maxIndex;
}

/*
 *
 */
double ReinforcementLearning::getMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return t_array[maxIndex];
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
	if(sum == 0) return rand()%t_array.size();
	double randomValue = ((double)(rand()%((int)100000000)))/100000000;
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
