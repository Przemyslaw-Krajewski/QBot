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
	alpha = 0.55;
	gamma = 0.80;

	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

	actions = nullptr;
	resetActionsNN();
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
void QLearning::resetActionsNN()
{
	if(actions != nullptr) delete actions;
	actions = new NeuralNetwork(dimensionStatesSize.size(),std::initializer_list<int>({120,100,numberOfActions}),
				std::initializer_list<double>({0.033,0.15,0.5}),1.2);
}

/*z
 *
 */
std::pair<bool,int> QLearning::chooseAction(State& t_state, ControlMode mode)
{
	std::vector<double> values;
	if(mode == ControlMode::NN)		  values = actions->determineY(convertState2NNInput(t_state));
	else if (mode == ControlMode::QL) values = qValues.getValues(t_state);
	else if (mode == ControlMode::Hybrid)
	{
		if(qValues.getChange(t_state) > ACTION_LEARN_THRESHOLD) values = actions->determineY(convertState2NNInput(t_state));
		else values = qValues.getValues(t_state);
	}
	else assert("no such control mode" && 0);

	int action = getIndexOfMaxValue(values);

	return std::pair<bool,int>(true,action);
}

/*
 *
 */
double QLearning::learnQL(State t_prevState, State t_state, int t_action, double t_reward)
{
	double maxValue = qValues.getValue(t_state,0);
	for(int i_action=1; i_action<numberOfActions; i_action++)
	{
		double value = qValues.getValue(t_state,i_action);
		if(maxValue < value) maxValue = value;
	}

	double prevValue = qValues.getValue(t_prevState,t_action);
	double value = prevValue + alpha*(t_reward+gamma*maxValue - prevValue);
	qValues.setValue(t_prevState, t_action, value);

	return qValues.getChange(t_prevState);
}

/*
 *
 */
std::pair<double,int> QLearning::learnAction(State state)
{

	if(qValues.getChange(state) > ACTION_LEARN_THRESHOLD) return std::pair<double,int>(0,2);

	std::vector<double> qlValues = qValues.getValues(state);
	std::vector<double> nnValues = actions->determineY(convertState2NNInput((state)));
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

	return std::pair<double,int>(err, qlAction==nnAction?1:0);
}

/*
 *
 */
NNInput QLearning::convertState2NNInput(State t_state)
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
