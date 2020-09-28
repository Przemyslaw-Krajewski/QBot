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
ActorCritic::ActorCritic(int t_nActions, int t_dimensionStatesSize) :
	criticValues(HashMap(t_nActions)),actorValues(HashMap(t_nActions))
{
	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;
}

/*
 *
 */
int ActorCritic::chooseAction(State& t_state)
{

	std::vector<double> critic = criticValues.getValues(t_state);
	std::vector<double> values = actorValues.getValues(t_state);

//	std::cout << critic[0] << "   :    ";
	double sum = 0;
	for(int i=0; i<values.size(); i++)
	{
//		std::cout << values[i] << "  ";
		sum += values[i];
	}
//	std::cout << "\n";
	return 0;

	if(sum == 0) return rand()%numberOfActions;
	double randomValue = ((double)(rand()%((int)100000)))/100000;
	for(int i=0; i<values.size(); i++)
	{
		randomValue -= (values[i]/sum);
		if(randomValue < 0)	return i;
	}

	return values.size()-1;
}

/*
 *
 */
double ActorCritic::learnSARS(State &t_prevState, State &t_state, int t_action, double t_reward)
{
	if(t_prevState.size() == 0 || t_reward == 0)
	{
		std::cout << t_prevState.size() << "  " << t_reward << "INVALID STATE!\n";
		return 0;
	}

	//Critic
	std::vector<double> values = criticValues.getValues(t_state);
//	double maxValue = getMaxValue(values);
//	double prevValue = criticValues.getValue(t_prevState,t_action);

//	double newValue = prevValue + ALPHA_PARAMETER*(t_reward+GAMMA_PARAMETER*maxValue - prevValue);
//	criticValues.setValue(t_prevState, t_action, newValue);
	criticValues.setValue(t_prevState, 0, t_reward);
	std::cout << values[0] << " -> " << t_reward << "\n";

	//Actor
	std::vector<double> statePrevValues = criticValues.getValues(t_prevState);
	std::vector<double> stateValues = criticValues.getValues(t_state);
	std::vector<double> actorZ = actorValues.getValues(t_prevState);



	double newActorValue = -(stateValues[0]-statePrevValues[0])*log(actorZ[t_action]) + actorZ[t_action];
//	std::cout << actorZ[t_action] << "  " << stateValues[0]-statePrevValues[0] << " " << newActorValue << "\n";
	actorValues.setValue(t_prevState,t_action,newActorValue);

	actorZ = actorValues.getValues(t_prevState);
	double sum = 0;
	for(int i=0; i<actorZ.size(); i++)
	{
		if(actorZ[i]> 0.999) actorZ[i] = 0.999;
		if(actorZ[i]< 0.001) actorZ[i] = 0.001;
	}
	for(int i=0; i<actorZ.size(); i++) {sum += actorZ[i];}
	for(int i=0; i<actorZ.size(); i++) actorZ[i] = actorZ[i]/sum;
	for(int i=0; i<actorZ.size(); i++) actorValues.setValue(t_prevState,i,actorZ[i]);

	return t_reward - values[0];
}

/*
 *
 */
double ActorCritic::learnFromScenario(std::list<SARS> &t_history)
{
	std::vector<SARS*> sarsPointers;
	sarsPointers.clear();
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = t_history.begin(); sarsIterator!=t_history.end(); sarsIterator++)
	{
		cumulatedReward = ActorCritic::LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
		sarsIterator->reward = cumulatedReward;
		sarsPointers.push_back(&(*sarsIterator));
//		double value = getCriticValue((sarsIterator)->oldState);
//		std::cout << sarsIterator->reward << "  " << value << "\n";
	}

	long counter=0;
	std::random_shuffle(sarsPointers.begin(),sarsPointers.end());

	//Learning
	double sumErr = 0;
	for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
	{
		sumErr += abs(learnSARS((*sarsIterator)->oldState,
								(*sarsIterator)->state,
								(*sarsIterator)->action,
								(*sarsIterator)->reward));
	}

	return sumErr;
}

/*
 *
 */
double ActorCritic::learnFromMemory()
{
	return 0;
}
