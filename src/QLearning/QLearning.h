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

#include "../Arrays/Array.h"
#include "../Arrays/Table.h"
#include "../Arrays/HashMapArray.h"
#include "../Arrays/NeuralNetworkArray.h"

#include "../ImageAnalyzer/ImageAnalyzer.h"

using State = std::vector<int>;

enum ValueMap {table, hashmap};

class QLearning {
public:
	QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize, ValueMap t_valueMap);
	virtual ~QLearning();

	double learn(State t_prevState, State t_state, int t_action, double t_reward);
	int chooseAction(State t_state);
	void learnActions();

	void setQValue(State t_state, int t_action, double t_value) {qValues->setValue(t_state, t_action, t_value);}
	double getQValue(State t_state, int t_action) { return qValues->getValue(t_state,t_action);}

	void printArrayInfo() {qValues->printInfo();}

	void addDiscoveredState(State t_state) {discoveredStates.insert(t_state);}

private:
	double gamma;
	double alpha;
	int numberOfActions;
	std::vector<int> dimensionStatesSize;

	HashMapArray *qValues;
	Array *actions;
	std::set<State> discoveredStates;


};

#endif /* SRC_QLEARNING_QLEARNING_H_ */
