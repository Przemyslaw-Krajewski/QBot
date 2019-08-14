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
	actorValues = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({500,400,300,numberOfActions}),
				std::initializer_list<double>({0.01,0.033,0.1,0.33}),0.8);
	if(criticValues != nullptr) delete criticValues;
	criticValues = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({500,400,300,1}),
				std::initializer_list<double>({0.01,0.033,0.1,0.33}),0.8);
}

/*
 *
 */
std::pair<bool,int> ActorCritic::chooseAction(State& t_state, ControlMode mode)
{
	std::vector<double> values = actorValues->determineY(t_state);
//		for(int i=0 ;i<values.size() ;i++) std::cout << values[i] << " ";
//		std::cout << "\n";
	int action = getIndexOfMaxValue(values);

	return std::pair<bool,int>(true,action);
}

/*
 *
 */
double ActorCritic::learn(State t_prevState, State t_state, int t_action, double t_reward)
{
	//Critic
	std::vector<double> prevStateValue = criticValues->determineY(t_prevState);
	std::vector<double> criticZ = std::vector<double>({t_reward});
	criticValues->learnBackPropagation(criticZ);

	//Actor
	std::vector<double> stateValue = criticValues->determineY(t_state);
	actorValues->determineY(t_prevState);
	std::vector<double> actorZ;
	if(stateValue[0] > prevStateValue[0])
	{
		for(int i=0 ; i<numberOfActions ; i++) actorZ.push_back(0.1);
		actorZ[t_action] = 0.9;
	}
	else
	{
		for(int i=0 ; i<numberOfActions ; i++) actorZ.push_back(0.9);
		actorZ[t_action] = 0.1;
	}
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
