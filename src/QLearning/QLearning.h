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

#include "../Game/Point.h"
#include "../Arrays/Array.h"
#include "../Arrays/Table.h"
#include "../Arrays/NeuralNetworkArray.h"
#include "../Arrays/HybridArray.h"
#include "../Arrays/HashMapArray.h"

using State = std::vector<int>;

enum ValueMap {table, neuralNetwork, hybrid, hashmap};

class QLearning {
public:
	QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize, ValueMap t_valueMap);
	virtual ~QLearning();

	double learn(State t_prevState, State t_state, int t_action, double t_reward);
	int chooseAction(State t_state);

	void setQValue(State t_state, int t_action, double t_value) {qValues->setValue(t_state, t_action, t_value);}
	double getQValue(State t_state, int t_action) { return qValues->getValue(t_state,t_action);}

	void printValuesMap(State state, std::string string);
	void printArrayInfo() {qValues->printInfo();}

private:
	double gamma;
	double alpha;
	int numberOfActions;

	Array *qValues;
};

#endif /* SRC_QLEARNING_QLEARNING_H_ */
