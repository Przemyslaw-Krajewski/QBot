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
QLearning::QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize) :
	qValues(QValues(t_nActions,t_dimensionStatesSize))
{
	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

	actions = nullptr;
	nnQValues = nullptr;
	resetActionsNN();
}

/*
 *
 */
QLearning::~QLearning()
{
	if(actions != nullptr) delete actions;
	if(nnQValues != nullptr) delete nnQValues;
}

/*
 *
 */
void QLearning::resetActionsNN()
{
	if(actions != nullptr) delete actions;
//	actions = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({400,380,300,numberOfActions}),
//				std::initializer_list<double>({0.01,0.033,0.1,0.33}),0.1);
	if(nnQValues != nullptr) delete nnQValues;
//	nnQValues = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({500,400,300,numberOfActions}),
//				std::initializer_list<double>({0.0033,0.01,0.033,0.1}),0.03);
}

/*
 *
 */
std::pair<bool,int> QLearning::chooseAction(State& t_state, ControlMode mode)
{
	std::vector<double> values = nnQValues->determineOutput(t_state);
//	for(int i=0 ;i<values.size() ;i++) std::cout << values[i] << " ";
//	std::cout << "\n";
	int action = getIndexOfMaxValue(values);

	return std::pair<bool,int>(true,action);
}

/*
 *
 */
double QLearning::learnQL(State t_prevState, State t_state, int t_action, double t_reward)
{
	std::vector<double> values = nnQValues->determineOutput(t_state);
	double maxValue = getMaxValue(values);

	std::vector<double> prevValues = nnQValues->determineOutput(t_prevState);
	double prevValue = prevValues[t_action];
	double newValue = prevValue + ALPHA_PARAMETER*(t_reward+GAMMA_PARAMETER*maxValue - prevValue);

//	std::cout << prevValues[t_action] << "  " << t_reward << "      " << prevValues[t_action]-t_reward << "\n";

	if (prevValues[t_action] < 0.5)
	{
		for(int i=0;i<numberOfActions;i++)
		{
			if(prevValues[i] < 0.7)	prevValues[i] += 0.005;
		}
	}
	prevValues[t_action] = newValue;

	nnQValues->learnBackPropagation(prevValues);

	std::vector<double> newPrevValues = nnQValues->determineOutput(t_prevState);

	return newPrevValues[t_action] - prevValues[t_action];
}

/*
 *
 */
std::pair<double,int> QLearning::learnAction(const State *state, bool skipNotReady)
{
//	int64 timeBefore = cv::getTickCount();
	if(qValues.getChange(*state) > ACTION_LEARN_THRESHOLD && skipNotReady) return std::pair<double,int>(0,2);

	std::vector<double> qlValues = qValues.getValues(*state);
	std::vector<double> nnValues = actions->determineOutput(*state);
	int qlAction = getIndexOfMaxValue(qlValues);
	int nnAction = getIndexOfMaxValue(nnValues);

	if(qlAction == nnAction) return std::pair<double,int>(0,3);

	std::vector<double> z;
	for(int i=0; i<numberOfActions ;i++) z.push_back(0.1);
	z[qlAction] = 0.9;

	actions->learnBackPropagation(z);

	//Calculate error;
	double err = 0;
	for(int i=0; i<nnValues.size(); i++) err += fabs(nnValues[i]-z[i]);

//	int64 timeAfter = cv::getTickCount();
//	std::cout << (timeAfter - timeBefore)/ cv::getTickFrequency() << "\n";

	return std::pair<double,int>(err, qlAction==nnAction?1:0);
}

/*
 *
 */
NNInput QLearning::convertState2NNInput(const State &t_state)
{
	NNInput result;
	for(int i=0; i<t_state.size(); i++) result.push_back((double) t_state[i]);
	return result;
}

/*
 *
 */
int QLearning::getIndexOfMaxValue(std::vector<double> t_array)
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
double QLearning::getMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return t_array[maxIndex];
}
