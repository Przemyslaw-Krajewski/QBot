/*
 * Bot.h
 *
 *  Created on: 13 lut 2019
 *      Author: mistrz
 */

#ifndef SRC_BOT_H_
#define SRC_BOT_H_

//#define PRINT_PROCESSING_TIME

#include <vector>
#include <list>
#include <math.h>
#include <iostream>
#include <fstream>

#include "../ActorCritic/ActorCritic.h"
#include "Common.h"
#include "../Analyzers/StateAnalyzer.h"

#include "../Analyzers/MemoryAnalyzer.h"
#include "../QLearning/QLearning.h"
#include "../ActorCritic/ActorCritic.h"
#include "../Loggers/DataDrawer.h"

class Bot {
	using ScenarioResult = StateAnalyzer::AnalyzeResult::AdditionalInfo;

public:
	Bot();
	virtual ~Bot();

	void execute();
	void testStateAnalyzer();

private:
	void loadParameters();

	void learnFromScenarioAC(std::list<SARS> &historyScenario);

	void learnFromMemoryAC();
	void eraseInvalidLastStates(std::list<SARS> &history);

	ControllerInput determineControllerInput(int t_action);
	int determineControllerInputInt(int t_action);
	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extractSceneState(std::vector<int> sceneData);
	static State reduceSceneState(const State& t_state, double action);

private:
	//
	StateAnalyzer analyzer;
	ActorCritic *actorCritic;

	std::map<ReducedState, SARS> memorizedSARS;
	int playsBeforeNNLearning;

	ControlMode controlMode;
	bool reset;

	//Const parameters
	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE= 0;

	const int TIME_LIMIT = 60;
	const int LEARN_FROM_HISTORY_ITERATIONS = 1;
	const int LEARN_FROM_MEMORY_ITERATIONS  = 0;
	const int PLAYS_BEFORE_NEURAL_NETWORK_LEARNING =0;
};

#endif /* SRC_BOT_H_ */
