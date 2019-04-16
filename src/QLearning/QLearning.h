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

#include "../Flags.h"
#include "../Bot/Common.h"

#include "../NeuralNetwork/NeuralNetwork.h"

enum ValueMap {table, hashmap};

class QLearning {

public:
	QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize, ValueMap t_valueMap);
	virtual ~QLearning();

	//Basic methods
	double learn(State t_prevState, State t_state, int t_action, double t_reward);
	int chooseAction(State t_state);
	//Extended methods
	void persistNN() {delete target; target = new NeuralNetwork(qValues);}

	//Info methods
	double getQValue(int t_action) {return qValues.getY()[t_action];}
	std::vector<double> getQValues(State t_state);
	void printArrayInfo() {/*DO NOTHING*/}

private:
	//Log methods
	void logNewSetMessage();
	void logLearningCompleteMessage();

public:
	//Debug methods
//	void setQValue(State t_state, int t_action, double t_value) {qValues->setValue(t_state, t_action, t_value);}
//	double getQValue(State t_state, int t_action) { return qValues->getValue(t_state,t_action);}


private:
	double gamma;
	double alpha;
	int numberOfActions;
	std::vector<int> dimensionStatesSize;

	NeuralNetwork qValues;
	NeuralNetwork* target;
};

#endif /* SRC_QLEARNING_QLEARNING_H_ */
