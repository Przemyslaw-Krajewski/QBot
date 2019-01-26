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
QLearning::QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize, ValueMap t_valueMap)
{

	alpha = 0.85;
	gamma = 0.90;

//	if(t_valueMap == table) qValues = new Table(t_nActions,t_dimensionStatesSize);
//	else if(t_valueMap == hashmap) qValues = new HashMapArray(t_nActions,t_dimensionStatesSize);

	qValues = new HashMapArray(t_nActions,t_dimensionStatesSize);


	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;
	actions = new NeuralNetworkArray(t_nActions,t_dimensionStatesSize);
}

/*
 *
 */
QLearning::~QLearning()
{
	delete qValues;
	delete actions;
}


/*
 *
 */
double QLearning::learn(State t_prevState, State t_state, int t_action, double t_reward)
{
	double maxValue = -999;
	for(int i_action=0; i_action<numberOfActions; i_action++)
	{
		double value = qValues->getValue(t_state,i_action);
		if(maxValue < value) maxValue = value;
	}

	double prevValue = qValues->getValue(t_prevState,t_action);
	double value = prevValue + alpha*(t_reward+gamma*maxValue - prevValue);
	qValues->setValue(t_prevState, t_action, value);

	return value - prevValue;
}

/*
 *
 */
std::pair<bool,int> QLearning::chooseAction(State t_state)
{
	std::vector<double> nnValues = actions->getValues(t_state);
	double nnMaxValue = -999;
	int nnAction;
	for(int i=0; i<numberOfActions; i++)
	{
		if(nnMaxValue < nnValues[i])
		{
			nnMaxValue = nnValues[i];
			nnAction = i;
		}
	}
	return std::pair<bool,int>(true,nnAction);

//	std::vector<double> qlValues = qValues->getValues(t_state);
//	double qlMaxValue = -999;
//	int qlAction;
//	for(int i=0; i<numberOfActions; i++)
//	{
//		if(qlMaxValue < qlValues[i])
//		{
//			qlMaxValue = qlValues[i];
//			qlAction = i;
//		}
//	}
//
//	if(qlValues[qlAction] > qlValues[nnAction]+150)
//	{
//		return std::pair<bool,int>(false,qlAction);
//	}
//	else return std::pair<bool,int>(true,nnAction);
}

/*
 *
 */

void QLearning::learnActions()
{
	std::cout << "I feel sleepy " << discoveredStates.size() << "\n";
	if(discoveredStates.size() <= 0) return;
	std::vector<const State*> shuffledStates;
	for(std::set<State>::iterator i=discoveredStates.begin(); i!=discoveredStates.end(); i++)
	{
		const State* s = &(*i);
		shuffledStates.push_back(s);
	}

	NeuralNetworkArray* newNN = new NeuralNetworkArray(numberOfActions,dimensionStatesSize);

	//Learn NN
	for(int iteration=0; iteration<discoveredStates.size()*20; iteration++)
	{
		double sumMistakes = 0;
		double sumErr = 0;
		double mistakeSumErr = 0;
		std::random_shuffle(shuffledStates.begin(),shuffledStates.end());
		logNewSetMessage();
		for(int j=0; j<shuffledStates.size(); j++)
		{
			const std::vector<int> randomState = *(shuffledStates[j]);
			std::vector<double> qValue = qValues->getValues(randomState);

			double maxValue = -999;
			int qlAction;
			for(int i=0; i<qValue.size(); i++)
			{
				if(qValue[i] > maxValue)
				{
					maxValue = qValue[i];
					qlAction = i;
				}
			}

			std::vector<double> values;
			for(int i=0; i<numberOfActions ;i++)
			{
				if(i==qlAction) values.push_back(0.9);
				else values.push_back(0.1);
			}

			std::vector<double> prevValues = newNN->setValues(randomState,values);

			maxValue = -999;
			int nnAction;
			double err = 0;
			for(int i=0; i<prevValues.size(); i++)
			{
				if(prevValues[i] > maxValue)
				{
					maxValue = prevValues[i];
					nnAction = i;
				}
				err += fabs(prevValues[i]-values[i]);
			}

			sumMistakes += (nnAction==qlAction ? 0 : 1);
			sumErr += err;
			if(nnAction!=qlAction) mistakeSumErr += err;
		}
		std::cout << "NN: "<< sumMistakes << "/" << discoveredStates.size() << "-" << iteration*0.1  << ":" << iteration << "   -   " << sumErr/(4*discoveredStates.size()) << ":" << mistakeSumErr/(4*sumMistakes) << "\n";
		if(sumMistakes <= iteration*0.1) break;
	}

	logLearningCompleteMessage();
	delete actions;
	actions = newNN;
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
