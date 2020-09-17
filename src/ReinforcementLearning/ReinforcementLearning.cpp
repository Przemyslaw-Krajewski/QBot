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

///*
// *
// */
//NNInput ReinforcementLearning::convertState2NNInput(const State &t_state)
//{
//	NNInput result;
//	for(int i=0; i<t_state.size(); i++) result.push_back((double) t_state[i]);
//	return result;
//}

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
	int reduceLevel = 4;
	std::vector<int> result;
	for(int i=0;i<t_state.size()-10;i++)
	{
		if(i%reduceLevel!=0) continue;
		result.push_back(t_state[i]);
	}
	for(int i=t_state.size()-10;i<t_state.size();i++)
	{
		result.push_back(t_state[i]);
	}
	result.push_back(action);

	return result;
}
