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
QLearning::QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize, ValueMap t_valueMap) :
	qValues(NeuralNetwork(t_dimensionStatesSize.size(),std::initializer_list<int>({80,70,60,50,t_nActions}),
			std::initializer_list<double>({0.000032,0.0001,0.00032,0.001,0.0032,}),6.2))
{
	target = new NeuralNetwork(qValues);
	alpha = 1;
	gamma = 0.80;
	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;
}

/*
 *
 */
QLearning::~QLearning()
{
	delete target;
}

/*
 *
 */
double QLearning::learn(State t_prevState, State t_state, int t_action, double t_reward)
{
	std::vector<double> state;
	for(int i=0; i<t_state.size();i++) state.push_back((double)t_state[i]/dimensionStatesSize[i]);

	std::vector<double> tv = target->determineY(state);
	double maxValue = -999;
	for(int i_action=0; i_action<numberOfActions; i_action++)
	{
		if(maxValue < tv[i_action]) maxValue = tv[i_action];
	}

	std::vector<double> qv = qValues.determineY(state);
	double prevValue = qv[t_action];
	double value = prevValue + alpha*(t_reward+gamma*maxValue - prevValue);

	std::vector<double> z = qv;
	z[t_action] = value;
	for(int i=0; i<z.size(); i++) if(z[i]>0.7) z[i]=0.61;
//	std::cout << prevValue << " --> " << value << "   (" << t_reward << " + " << gamma << "*" << maxValue << ")\n";
	qValues.learnBackPropagation(z);

	return value - prevValue;
}

/*
 *
 */
int QLearning::chooseAction(State t_state)
{
	std::vector<double> state;
	for(int i=0; i<t_state.size();i++) state.push_back((double)t_state[i]/dimensionStatesSize[i]);

	std::vector<double> nnValues = qValues.determineY(state);
	double nnMaxValue = -999;
	int nnAction;
	for(int i=0; i<numberOfActions; i++)
	{
		if(nnValues[i] > 0.61 || nnValues[i] < 0.0001) {return i;}
		if(nnMaxValue < nnValues[i])
		{
			nnMaxValue = nnValues[i];
			nnAction = i;
		}
	}
	return nnAction;
}

/*
 *
 */
std::vector<double> QLearning::getQValues(State t_state)
{
	std::vector<double> input;
	for(int i=0; i<t_state.size(); i++) input.push_back((double)t_state[i]);
	return qValues.determineY(input);
}

void QLearning::logNewSetMessage()
{
	#ifdef ENABLE_LOGGING
		std::cerr << "=============================================\nNEW SET\n==================================================\n";
	#endif
}

void QLearning::logLearningCompleteMessage()
{
	#ifdef ENABLE_LOGGING
		std::cerr << "=============================================\nLEARNING COMPLETE\n==================================================\n";
	#endif
}
