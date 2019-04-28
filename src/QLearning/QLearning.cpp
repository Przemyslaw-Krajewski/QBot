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
	gamma = 0.80;

//	if(t_valueMap == table) qValues = new Table(t_nActions,t_dimensionStatesSize);
//	else if(t_valueMap == hashmap) qValues = new HashMapArray(t_nActions,t_dimensionStatesSize);

	qValues = new HashMapArray(t_nActions,t_dimensionStatesSize);
	qChanges = new HashMapArray(t_nActions,t_dimensionStatesSize);

	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;
	actions = new NeuralNetworkArray(t_nActions,t_dimensionStatesSize);
}

/*
 *
 */
QLearning::~QLearning()
{
	if(qValues!=nullptr) delete qValues;
	if(actions!=nullptr) delete actions;
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
	qChanges->setValue(t_prevState, 0, abs(value-prevValue));

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

	int skip = sqrt(shuffledStates.size())/2;
	skip = skip<1 ? 1 : skip;
	//Learn NN
	for(int iteration=0; iteration<1; iteration++)
	{
		double sumErr = 0;
		std::random_shuffle(shuffledStates.begin(),shuffledStates.end());
		for(int j=0; j<shuffledStates.size(); j+=skip)
		{
			const std::vector<int> randomState = *(shuffledStates[j]);
			double err = learnAction(randomState);
			sumErr += err;
		}
		std::cout << "NN: "<< sumErr << "/" << discoveredStates.size() << "\n" ;
	}
}

/*
 *
 */
double QLearning::learnAction(State state)
{
	if(qChanges->getValue(state,0) > 3) return 0;

	std::vector<double> qValue = qValues->getValues(state);
	std::vector<double> prevValues = actions->getValues(state);

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

	maxValue = -999;
	int nnAction;
	for(int i=0; i<prevValues.size(); i++)
	{
		if(prevValues[i] > maxValue)
		{
			maxValue = prevValues[i];
			nnAction = i;
		}
	}

	if(qlAction == nnAction) return 0;

	std::vector<double> values;
	for(int i=0; i<numberOfActions ;i++)
	{
		if(i==qlAction) values.push_back(0.9);
		else values.push_back(0.1);
	}

	actions->setValues(state,values);

	maxValue = -999;
	//int nnAction;
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
	return err;
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
