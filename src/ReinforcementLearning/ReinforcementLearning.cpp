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
State ReinforcementLearning::reduceSceneState(const State& t_state, double action)
{
	int reduceLevel = 8;
	int xSize = 32;
	int ySize = 56;

	std::vector<int> result;
	for(int x=0;x<32;x+=reduceLevel)
	{
		for(int y=0;y<56;y+=reduceLevel)
		{
			int value=0;
			for(int xx=0;xx<reduceLevel;xx++)
			{
				for(int yy=0;yy<reduceLevel;yy++)
				{
					if(t_state[(x+xx)*56+y+yy] > value) value = t_state[+(x+xx)*56+y+yy];
					if(t_state[32*56+(x+xx)*56+y+yy]*2 > value) value = t_state[32*56+(x+xx)*56+y+yy]*2;
				}
			}
			result.push_back(value);
		}
	}
	result.push_back(t_state[t_state.size()-8]/3);
	result.push_back(t_state[t_state.size()-9]);
	result.push_back(action);

	return result;
}
