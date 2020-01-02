/*
 * QLearning.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_ACTORCRITIC_ACTORCRITIC_H_
#define SRC_ACTORCRITIC_ACTORCRITIC_H_

#include <vector>
#include <string>

#include "../Bot/Common.h"
#include "../HashMap/QValues.h"
#include "../NeuralNetwork/NeuralNetwork.h"

class ActorCritic {

public:
	ActorCritic(int t_nActions, std::vector<int> t_dimensionStatesSize);
	virtual ~ActorCritic() = default;

	//Basic methods
	std::pair<bool,int> chooseAction(State& t_state, ControlMode mode);
	double learn(State t_prevState, State t_state, int t_action, double t_reward);

	void resetNN();

	double getCriticValue(State t_state) {return criticValues.determineOutput(t_state)[0];}


private:
	NNInput convertState2NNInput(const State &t_state);
	int getIndexOfMaxValue(std::vector<double> t_array);
	double getMaxValue(std::vector<double> t_array);

public:
	//Debug methods
//	void setQValue(State t_state, int t_action, double t_value) {qValues.setValue(t_state, t_action, t_value);}
//	double getQValue(State t_state, int t_action) { return qValues.getValue(t_state,t_action);}
//	double getQChange(State t_state) { return qValues.getChange(t_state);}


private:
	int numberOfActions;
	std::vector<int> dimensionStatesSize;

	NeuralNetwork criticValues;
	NeuralNetwork actorValues;

public:
	static constexpr double ACTION_LEARN_THRESHOLD = 40;

	static constexpr double GAMMA_PARAMETER = 0;		//reward for advancing to next promising state
	static constexpr double ALPHA_PARAMETER = 1;		//speed of learning QLearning
	static constexpr double LAMBDA_PARAMETER = 0.9;		//reward cumulation factor


};

#endif /* SRC_ACTORCRITIC_ACTORCRITIC_H */
