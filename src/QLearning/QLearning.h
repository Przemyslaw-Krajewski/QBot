/*
 * QLearning.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_QLEARNING_QLEARNING_H_
#define SRC_QLEARNING_QLEARNING_H_

#include <vector>
#include <string>

#include "../HashMap/HashMap.h"
#include "../NeuralNetwork/NeuralNetwork.h"

using State = std::vector<int>;

class QLearning {

public:
	QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize);
	virtual ~QLearning();

	//Basic methods
	std::pair<bool,int> chooseAction(State t_state);
	double learnQL(State t_prevState, State t_state, int t_action, double t_reward);
	double learnAction(State state);


private:
	NNInput convertState2NNInput(State t_state);
	int getIndexOfMaxValue(std::vector<double> t_array);

public:
	//Debug methods
	void setQValue(State t_state, int t_action, double t_value) {qValues.setValue(t_state, t_action, t_value);}
	double getQValue(State t_state, int t_action) { return qValues.getValue(t_state,t_action);}


private:
	double gamma;
	double alpha;
	int numberOfActions;
	std::vector<int> dimensionStatesSize;

	HashMap qValues;
	NeuralNetwork actions;


};

#endif /* SRC_QLEARNING_QLEARNING_H_ */
