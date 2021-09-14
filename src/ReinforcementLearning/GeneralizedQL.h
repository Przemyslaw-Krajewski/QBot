/*
 * QLearning.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_REINFORCEMENTLEARN_GENQLEARNING_H_
#define SRC_REINFORCEMENTLEARN_GENQLEARNING_H_

#include <vector>
#include <string>

#include "ReinforcementLearning.h"
#include "../Bot/Common.h"
#include "../HashMap/HashMap.h"
#include "../NeuralNetwork/NeuralNetwork.h"
#include "../NeuralNetworkGPU/NeuralNetwork.h"

class GeneralizedQL : public ReinforcementLearning
{

public:
	GeneralizedQL(int t_nActions, int t_dimensionStatesSize);
	virtual ~GeneralizedQL();

	//Basic methods
	virtual int chooseAction(State& t_state) override;
	virtual double learnSARS(SARS &t_sars) override;

	virtual double learnFromScenario(std::list<SARS> &t_history) override;
	virtual double learnFromMemory() override;

	std::pair<double,int> learnAction(State &state, bool skipNotReady = true);



	//NeuralNetwork methods
	void resetActionsNN();

	//File operation
//	void saveQValues() {qValues.saveToFile();}
//	void loadQValues() {qValues.loadFromFile();}
//	void saveNeuralNetwork() {actions->saveToFile();}
//	void loadNeuralNetwork() {actions->loadFromFile();}
//	std::vector<State> getStateList() {return qValues.getStateList();}

public:
	//Debug methods
	void setQValue(State t_state, int t_action, double t_value) {qValues.setValue(t_state, t_action, t_value);}
	double getQValue(State t_state, int t_action) { return qValues.getValue(t_state,t_action);}
	double getQChange(State t_state) { return qValues.getChange(t_state);}


private:
	double gamma;				//+% of next state reward
	double alpha;				//How fast is learning QL
	int numberOfActions;
	int dimensionStatesSize;

	HashMap qValues;
	NeuralNetworkCPU::NeuralNetwork *actions;

	ControlMode controlMode;

public:
	static constexpr double ACTION_LEARN_THRESHOLD = 40;

	static constexpr double GAMMA_PARAMETER = 0.9;		//reward for advancing to next promising state
	static constexpr double ALPHA_PARAMETER = 0.75;		//speed of learning QLearning
	static constexpr double LAMBDA_PARAMETER = 0.0;		//reward cumulation factor

	static constexpr int LEARN_FROM_HISTORY_ITERATIONS = 2;
	static constexpr int LEARN_FROM_MEMORY_ITERATIONS  = 1;

};

#endif /* SRC_REINFORCEMENTLEARN_GENQLEARNING_H_ */
