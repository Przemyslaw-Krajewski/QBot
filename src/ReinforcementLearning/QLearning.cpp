/*
 * QLearning.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "QLearning.h"

/*
 *
 */
QLearning::QLearning(int t_nActions, int t_dimensionStatesSize) :
	qValues(HashMap(t_nActions))
{
	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;
}

/*
 *
 */
QLearning::~QLearning()
{

}

/*
 *
 */
int QLearning::chooseAction(State& t_state)
{
	std::vector<double> values = qValues.getValues(t_state);
	int action = getIndexOfMaxValue(values);

	return action;
}

/*
 *
 */
double QLearning::learnSARS(State &t_prevState, State &t_state, int t_action, double t_reward)
{
	std::vector<double> values = qValues.getValues(t_state);
	double maxValue = getMaxValue(values);
	double prevValue = qValues.getValue(t_prevState,t_action);

	double newValue = prevValue + ALPHA_PARAMETER*(t_reward+GAMMA_PARAMETER*maxValue - prevValue);
	qValues.setValue(t_prevState, t_action, newValue);

	return newValue- prevValue;
}

/*
 *
 */
double QLearning::learnFromScenario(std::list<SARS> &t_history)
{
	double change = 0;
	//QLearning
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = t_history.begin(); sarsIterator!=t_history.end(); sarsIterator++)
	{
		change += abs(learnSARS(sarsIterator->oldState,
				  			    sarsIterator->state,
								sarsIterator->action,
								sarsIterator->reward + cumulatedReward));
		cumulatedReward = LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
	}
	return change;
}

/*
 *
 */
double QLearning::learnFromMemory()
{
	return 0;
}
