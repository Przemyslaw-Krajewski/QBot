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
