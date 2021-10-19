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
double QLearning::learnSARS(SARS &t_sars)
{
	std::vector<double> values = qValues.getValues(t_sars.state);
	double maxValue = getMaxValue(values);
	double prevValue = qValues.getValue(t_sars.oldState,t_sars.action);

	double newValue = prevValue + ALPHA_PARAMETER*(t_sars.reward+GAMMA_PARAMETER*maxValue - prevValue);
	qValues.setValue(t_sars.oldState, t_sars.action, newValue);

	return newValue- prevValue;
}

/*
 *
 */
LearningStatus QLearning::learnFromScenario(std::list<SARS> &t_history)
{
	double sumErr = 0;
	//QLearning
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = t_history.begin(); sarsIterator!=t_history.end(); sarsIterator++)
	{
		sarsIterator->reward = sarsIterator->reward + cumulatedReward;
		sumErr += abs(learnSARS(*sarsIterator));
		cumulatedReward = LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
	}
	return LearningStatus(sumErr,0);
}

/*
 *
 */
LearningStatus QLearning::learnFromMemory()
{
	return LearningStatus(0,0);
}
