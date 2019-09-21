/*
 * QLearning.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "ActorCritic.h"

/*
 *
 */
ActorCritic::ActorCritic(int t_nActions, std::vector<int> t_dimensionStatesSize)
{
	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

	actorValues = nullptr;
	criticValues = nullptr;
	resetActionsNN();
}

/*
 *
 */
ActorCritic::~ActorCritic()
{
	if(actorValues != nullptr) delete actorValues;
	if(criticValues != nullptr) delete criticValues;
}

/*
 *
 */
void ActorCritic::resetActionsNN()
{
	if(actorValues != nullptr) delete actorValues;
	actorValues = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({100,80,numberOfActions}),
				std::initializer_list<double>({0.033,0.1,0.33}),1.2);
	if(criticValues != nullptr) delete criticValues;
	criticValues = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({100,80,1}),
				std::initializer_list<double>({0.033,0.1,0.33}),1.2);
}

/*
 *
 */
std::pair<bool,int> ActorCritic::chooseAction(State& t_state, ControlMode mode)
{
	std::vector<double> values = actorValues->determineY(t_state);
	double sum = 0;
	for(int i=0; i<values.size(); i++)
	{
		if(values[i] < 0.3) continue;
		sum += values[i];
	}
	double randomValue = ((double)(rand()%((int)sum*10000)))/10000;
	for(int i=0; i<values.size(); i++)
	{
		if(values[i] < 0.3) continue;
		randomValue -= values[i];
		if(randomValue < 0) return std::pair<bool,int>(true,i);
	}

	return std::pair<bool,int>(true,values.size()-1);
}

/*
 *
 */
double ActorCritic::learn(State t_prevState, State t_state, int t_action, double t_reward)
{
	//Critic
	std::vector<double> prevStateValue = criticValues->determineY(t_prevState);
	std::vector<double> criticZ = std::vector<double>();
	criticZ.push_back(t_reward);
	criticValues->learnBackPropagation(criticZ);

	//Actor
	std::vector<double> stateValue = criticValues->determineY(t_state);
	std::vector<double> actorZ = actorValues->determineY(t_prevState);

	actorZ[t_action] = 0.5*t_reward/actorZ[t_action];

//	if(prevStateValue[0] >= 0.6 || prevStateValue[0] < stateValue[0]) actorZ[t_action] = 0.9;
//	else actorZ[t_action] = 0.1;
//
//	for(int i=0 ; i<numberOfActions ; i++)
//	{
//		if(actorZ[i] < 0.1) actorZ[i] = 0.10;
//	}

//	if((chosenAction!=t_action && prevStateValue[0] >= 0.5) ||
//			(chosenAction==t_action && prevStateValue[0] < 0.5))
		actorValues->learnBackPropagation(actorZ);

	return prevStateValue[0] - t_reward;
}

/*
 *
 */
NNInput ActorCritic::convertState2NNInput(const State &t_state)
{
	NNInput result;
	for(int i=0; i<t_state.size(); i++) result.push_back((double) t_state[i]);
	return result;
}

/*
 *
 */
int ActorCritic::getIndexOfMaxValue(std::vector<double> t_array)
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
double ActorCritic::getMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return t_array[maxIndex];
}
