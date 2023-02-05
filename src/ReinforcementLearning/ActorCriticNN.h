/*
 * QLearning.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_ACTORCRITIC_ACTORCRITICNN_H_
#define SRC_ACTORCRITIC_ACTORCRITICNN_H_

#define ACTORCRITICNN_LOG_ACTOR
//#define ACTORCRITICNN_LOG_CRITIC
#define ACTORCRITICNN_LOG_NN
//#define ACTORCRITICNN_ONE_ACTION

#include <vector>
#include <string>

#include "ReinforcementLearning.h"
#include "../Bot/State.h"
#include "../HashMap/HashMap.h"
#include "../NeuralNetwork/NeuralNetwork.h"
#include "../NeuralNetworkGPU/NeuralNetwork.h"

#include "../Analyzers/ImageAnalyzer/ImageAnalyzer.h"

class ActorCriticNN : public ReinforcementLearning
{

public:
	ActorCriticNN(int t_nActions, int t_dimensionStatesSize, ReduceStateMethod t_reduceStateMethod);
	virtual ~ActorCriticNN();

	//Basic methods
	virtual int chooseAction(State& t_state) override;
	virtual double learnSARS(SARS &t_sars) override;

	virtual LearningStatus learnFromScenario(std::list<SARS> &t_history) override;
	virtual LearningStatus learnFromMemory() override;
	virtual void prepareScenario(std::list<SARS> &t_history, int additionalInfo) override;

	//Additional methods
	virtual void handleUserInput(char t_userInput);
	virtual std::map<State, SARS>* getMemoryMap() {return &memorizedSARS;}
	void resetNN();
	void createNN0();
	void createNNA();
	void createNNB();
	void createNNC();
	void createNND();

private:
	char processLearningFromSARS(std::vector<SARS*> t_sars);
	void putStateToMemory(SARS &t_sars);

private:
	int numberOfActions;
	int dimensionStatesSize;
	int imageSize;

	NeuralNetworkGPU::NeuralNetwork criticValues;
	NeuralNetworkGPU::NeuralNetwork actorValues;

	std::map<State, SARS> memorizedSARS;

	ReduceStateMethod reduceStateMethod;

public:
	static constexpr double UPPER_REWARD_CUP = 0.75;
	static constexpr double LOWER_REWARD_CUP = 0.15;
	static constexpr double MEMORIZE_SARS_CUP = 0.70;

	const int LEARN_FROM_HISTORY_ITERATIONS = 1;
	const int LEARN_FROM_MEMORY_ITERATIONS  = 1;

	static constexpr double GAMMA_PARAMETER = 0;		//reward for advancing to next promising state
	static constexpr double ALPHA_PARAMETER = 1;		//speed of learning QLearning
	static constexpr double LAMBDA_PARAMETER = 0.93;	//reward cumulation factor


};

#endif /* SRC_ACTORCRITIC_ACTORCRITICNN_H */
