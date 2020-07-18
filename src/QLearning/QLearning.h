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

#include "../Bot/Common.h"
#include "../HashMap/HashMap.h"
#include "../NeuralNetworkGPU/NeuralNetwork.h"

class QLearning {

public:
	QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize);
	virtual ~QLearning();

	//Basic methods
	std::pair<bool,int> chooseAction(State& t_state, ControlMode mode);
	std::pair<bool,int> chooseActorAction(State& t_state, ControlMode mode);
	double learnQL(State t_prevState, State t_state, int t_action, double t_reward);

	//NeuralNetwork methods
	void resetActionsNN();
	void copyQValuesToTarget();

	//File operation
	void saveQValues() {qValues->saveToFile();}
	void loadQValues() {qValues->loadFromFile();}

private:
	//Helping
	NeuralNetworkGPU::NNInput convertState2NNInput(const State &t_state);
	int getIndexOfMaxValue(std::vector<double> t_array);

public:
	//Debug methods
	double getQValue(State t_state, int t_action) { return qValues->determineOutput(t_state)[t_action];}


private:
	double gamma;				//+% of next state reward
	double alpha;				//How fast is learning QL
	int numberOfActions;
	std::vector<int> dimensionStatesSize;

	NeuralNetworkGPU::NeuralNetwork *qValues;
	NeuralNetworkGPU::NeuralNetwork *target;
	NeuralNetworkGPU::NeuralNetwork *actor;

public:
	static constexpr double ACTION_LEARN_THRESHOLD = 40;


};

#endif /* SRC_QLEARNING_QLEARNING_H_ */
