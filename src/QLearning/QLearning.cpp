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
	qValues(NeuralNetwork(t_dimensionStatesSize.size(),std::initializer_list<int>({80,70,t_nActions}),
			std::initializer_list<double>({0.0001,0.0032,0.001}),5.2))
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


    std::vector<double> prevState;
	for(int i=0; i<t_prevState.size();i++) prevState.push_back((double)t_prevState[i]/dimensionStatesSize[i]);
	std::vector<double> qv = qValues.determineY(prevState);
	double prevValue = qv[t_action];
//	std::cout << prevValue << " -> " << maxValue << "  " << t_reward <<  "\n";
	double value = prevValue + alpha*(t_reward+gamma*maxValue - prevValue);

	std::vector<double> z = qv;
	z[t_action] = value;
	for(int i=0; i<z.size(); i++) if(z[i]>0.7) z[i]=0.61;

	qValues.learnBackPropagation(z);

	return value - prevValue;
}

/*
 *
 */
double QLearning::learn(std::vector<SARS> t_sars)
{
    double change = 0;
    std::vector<double> qv;
    std::vector<double> oldState;
    for(int it_sars=0; it_sars<t_sars.size(); it_sars++)
    {
        if(t_sars[it_sars].action != -1)
        {
            for(int i=0; i<t_sars[it_sars].oldState.size();i++)
                oldState.push_back((double)t_sars[it_sars].oldState[i]/dimensionStatesSize[i]);
            qv = qValues.determineY(oldState);
            break;
        }
        oldState.clear();
    }
    std::vector<double> z = qv;
    for(int it_sars=0; it_sars<t_sars.size(); it_sars++)
    {
        if(t_sars[it_sars].action == -1) continue;
        std::vector<double> state;
        for(int i=0; i<t_sars[it_sars].state.size();i++)
            state.push_back((double)t_sars[it_sars].state[i]/dimensionStatesSize[i]);
        std::vector<double> tv = target->determineY(state);

        double targetMaxValue = -999;
        for(int i_action=0; i_action<numberOfActions; i_action++)
        {
            if(targetMaxValue < tv[i_action]) targetMaxValue = tv[i_action];
        }

        double prevValue = qv[t_sars[it_sars].action];
        double reward = t_sars[it_sars].reward;
        double value = prevValue + alpha*(reward+gamma*targetMaxValue - prevValue);

        z[t_sars[it_sars].action] = value;
        change += value - prevValue;

    }
    for(int i=0;i<z.size();i++) if(z[i]>0.7) z[i] = 0.61;
	qValues.learnBackPropagation(z);

	return change;
}

/*
 *
 */
std::pair<int,double> QLearning::chooseAction(State t_state)
{
	std::vector<double> state;
	for(int i=0; i<t_state.size();i++) state.push_back((double)t_state[i]/dimensionStatesSize[i]);

	std::vector<double> nnValues = qValues.determineY(state);
	double nnMaxValue = -999;
	int nnAction;
	for(int i=0; i<numberOfActions; i++)
	{
		if(nnValues[i] > 0.61 || nnValues[i] < 0.0001) {return std::pair<int,double>(i,nnValues[i]);}
		if(nnMaxValue < nnValues[i])
		{
			nnMaxValue = nnValues[i];
			nnAction = i;
		}
	}
	return std::pair<int,double>(nnAction,nnMaxValue);
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
