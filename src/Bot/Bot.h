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

#include "Common.h"
#include "../Analyzers/StateAnalyzer.h"

#include "../Analyzers/MemoryAnalyzer.h"
#include "../Loggers/DataDrawer.h"
#include "../ReinforcementLearning/QLearning.h"
#include "../ReinforcementLearning/GeneralizedQL.h"
#include "../ReinforcementLearning/ActorCriticNN.h"

class Bot {

public:
	Bot();
	virtual ~Bot();

	void execute();

private:
	void loadParameters();

	ControllerInput determineControllerInput(int t_action);
	int determineControllerInputInt(int t_action);
	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extractSceneState(std::vector<int> sceneData, int xScreenSize=32, int yScreenSize=56);

private:
	StateAnalyzer stateAnalyzer;
	ReinforcementLearning *reinforcementLearning;

	//Const parameters
	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE= 0;

	const int TIME_LIMIT = 150;
};

#endif /* SRC_BOT_H_ */
