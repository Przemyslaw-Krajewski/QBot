/*
 * QLearning.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_ACTORCRITIC_ACTORCRITICNN_H_
#define SRC_ACTORCRITIC_ACTORCRITICNN_H_

#define ACTORCRITICNN_LOG
//#define ACTORCRITICNN_ONE_ACTION

#include <vector>
#include <string>

#include "ReinforcementLearning.h"
#include "../Bot/Common.h"
#include "../HashMap/HashMap.h"
#include "../NeuralNetwork/NeuralNetwork.h"
#include "../NeuralNetworkGPU/NeuralNetwork.h"

#include "../Analyzers/StateAnalyzer.h"

#include "../Loggers/ParameterFIleHandler.h"

class ActorCriticNN : public ReinforcementLearning
{

public:
	ActorCriticNN(int t_nActions, int t_dimensionStatesSize, StateAnalyzer *t_stateAnalyzer);
	virtual ~ActorCriticNN();

	//Basic methods
	virtual int chooseAction(State& t_state) override;
	virtual double learnSARS(SARS &t_sars) override;

	virtual double learnFromScenario(std::list<SARS> &t_history) override;
	virtual double learnFromMemory() override;

	//Additional methods
	virtual void handleParameters();
	void resetNN();
	void createNN0();
	void createNNA();
	void createNNB();
	void createNNC();
	void createNND();

protected:
	double getExp(double value) {return exp(8*value);}

private:
	void processLearningFromSARS(std::vector<SARS*> t_sars);
	void putStateToMemory(SARS &t_sars);

private:
	int numberOfActions;
	int dimensionStatesSize;
	int imageSize;

	NeuralNetworkGPU::NeuralNetwork criticValues;
	NeuralNetworkGPU::NeuralNetwork actorValues;

	std::map<State, SARS> memorizedSARS;

	StateAnalyzer *stateAnalyzer;

public:
	static constexpr double UPPER_REWARD_CUP = 0.75;
	static constexpr double LOWER_REWARD_CUP = 0.15;
	static constexpr double MEMORIZE_SARS_CUP = 0.70;

	const int LEARN_FROM_HISTORY_ITERATIONS = 1;
	const int LEARN_FROM_MEMORY_ITERATIONS  = 1;

	static constexpr double GAMMA_PARAMETER = 0;		//reward for advancing to next promising state
	static constexpr double ALPHA_PARAMETER = 1;		//speed of learning QLearning
	static constexpr double LAMBDA_PARAMETER = 0.92;	//reward cumulation factor


};

#endif /* SRC_ACTORCRITIC_ACTORCRITICNN_H */
