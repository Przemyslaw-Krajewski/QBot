/*
 * QLearning.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_ACTORCRITIC_ACTORCRITICNN_H_
#define SRC_ACTORCRITIC_ACTORCRITICNN_H_

#include <vector>
#include <string>

#include "ReinforcementLearning.h"
#include "../Bot/Common.h"
#include "../HashMap/HashMap.h"
#include "../NeuralNetwork/NeuralNetwork.h"
#include "../NeuralNetworkGPU/NeuralNetwork.h"

class ActorCriticNN : public ReinforcementLearning
{

public:
	ActorCriticNN(int t_nActions, int t_dimensionStatesSize);
	virtual ~ActorCriticNN() = default;

	//Basic methods
	virtual int chooseAction(State& t_state) override;
	virtual double learnSARS(State &t_prevState, State &t_state, int t_action, double t_reward) override;

	virtual double learnFromScenario(std::list<SARS> &t_history) override;
	virtual double learnFromMemory() override;

	void resetNN();

	double getCriticValue(State t_state) {return criticValues.determineOutput(t_state)[0];}
//	void drawCriticValues() {criticValues.drawNeuralNetwork();}

public:
	//Debug methods
//	void setQValue(State t_state, int t_action, double t_value) {qValues.setValue(t_state, t_action, t_value);}
//	double getQValue(State t_state, int t_action) { return qValues.getValue(t_state,t_action);}
//	double getQChange(State t_state) { return qValues.getChange(t_state);}


private:
	int numberOfActions;
	int dimensionStatesSize;

	NeuralNetworkGPU::NeuralNetwork criticValues;
	NeuralNetworkGPU::NeuralNetwork actorValues;

	std::map<State, SARS> memorizedSARS;

public:
	const int LEARN_FROM_HISTORY_ITERATIONS = 1;
	const int LEARN_FROM_MEMORY_ITERATIONS  = 1;

	static constexpr double GAMMA_PARAMETER = 0;		//reward for advancing to next promising state
	static constexpr double ALPHA_PARAMETER = 1;		//speed of learning QLearning
	static constexpr double LAMBDA_PARAMETER = 0.92;		//reward cumulation factor


};

#endif /* SRC_ACTORCRITIC_ACTORCRITICNN_H */
