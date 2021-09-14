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

#include "ReinforcementLearning.h"

#include "../Bot/Common.h"
#include "../HashMap/HashMap.h"
#include "../NeuralNetwork/NeuralNetwork.h"

class QLearning : public ReinforcementLearning
{

public:
	QLearning(int t_nActions, int t_dimensionStatesSize);
	virtual ~QLearning();

	//Basic methods
	virtual int chooseAction(State& t_state) override;
	virtual double learnSARS(SARS &t_sars) override;

	virtual double learnFromScenario(std::list<SARS> &t_history) override;
	virtual double learnFromMemory() override;

public:
	//Debug methods
	void setQValue(State t_state, int t_action, double t_value) {qValues.setValue(t_state, t_action, t_value);}
	double getQValue(State t_state, int t_action) { return qValues.getValue(t_state,t_action);}
	double getQChange(State t_state) { return qValues.getChange(t_state);}


private:
	int numberOfActions;
	int dimensionStatesSize;

	HashMap qValues;

public:
	static constexpr double ACTION_LEARN_THRESHOLD = 40;

	static constexpr double GAMMA_PARAMETER = 0.9;		//reward for advancing to next promising state
	static constexpr double ALPHA_PARAMETER = 0.75;		//speed of learning QLearning
	static constexpr double LAMBDA_PARAMETER = 0.4;		//reward cumulation factor


};

#endif /* SRC_QLEARNING_QLEARNING_H_ */
