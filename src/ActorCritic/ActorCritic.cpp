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
	resetCritic();
	resetActor();
}

/*
 *
 */
void ActorCritic::resetCritic()
{
	if(criticValues != nullptr) delete criticValues;
	criticValues = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({900,1}),
				std::initializer_list<double>({0.04,0.06}),1.1);
}

/*
 *
 */
void ActorCritic::resetActor()
{
	if(actorValues != nullptr) delete actorValues;
	actorValues = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({900,numberOfActions}),
				std::initializer_list<double>({0.04,0.06}),1.1);
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
		std::cout << values[i] << "  ";
		sum += values[i];
	}
	std::cout << "\n";

	if(sum == 0) return std::pair<bool,int>(true,rand()%numberOfActions);
	double randomValue = ((double)(rand()%((int)10000)))/10000;
	for(int i=0; i<values.size(); i++)
	{
		randomValue -= values[i]/sum;
		if(values[i] > 0.60 || randomValue < 0)	return std::pair<bool,int>(true,i);
	}

	return std::pair<bool,int>(true,values.size()-1);
}

/*
 *
 */
double ActorCritic::learn(State t_prevState, State t_state, int t_action, double t_reward)
{
	if(t_prevState.size() == 0 || t_reward == 0)
	{
		std::cout << "INVALID STATE!\n";
		return 0;
	}

	//Critic
	std::vector<double> prevStateValue = criticValues->determineY(t_prevState);
	std::vector<double> criticZ = std::vector<double>();
	criticZ.push_back(t_reward);
	criticValues->learnBackPropagation(criticZ);

	//Actor
	std::vector<double> stateValue = criticValues->determineY(t_state);
	std::vector<double> actorZ = actorValues->determineY(t_prevState);

	double sum = 0;
	for(int i=0 ; i<numberOfActions ; i++) sum += actorZ[i];
	actorZ[t_action] = -(prevStateValue[0]-stateValue[0])/actorZ[t_action] + actorZ[t_action];

	for(int i=0 ; i<numberOfActions ; i++)
	{
		if(actorZ[i] < 0.01) actorZ[i] = 0.01;
		if(actorZ[i] > 0.99) actorZ[i] = 0.99;
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
