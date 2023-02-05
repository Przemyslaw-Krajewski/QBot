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

#include "Controller.h"
#include "State.h"

#include "../Analyzers/StateAnalyzer/StateAnalyzer.h"

#include "../Analyzers/MemoryAnalyzer.h"
#include "../Loggers/DataDrawer.h"
#include "../Loggers/StateViewer.h"
#include "../Loggers/LogFileHandler.h"
#include "../ReinforcementLearning/QLearning.h"
#include "../ReinforcementLearning/GeneralizedQL.h"
#include "../ReinforcementLearning/ActorCriticNN.h"

class Bot {

public:
	Bot();
	virtual ~Bot();

	void execute();

private:
	int handleUserInput(char input);

private:
	StateAnalyzer stateAnalyzer;
	ReinforcementLearning *reinforcementLearning;

	StateAnalyzer::AnalyzeResult analyzeResult;
	StateAnalyzer::AnalyzeResult prevAnalyzeResult;
	Controller controller;
	Controller prevController;
	std::list<SARS> historyScenario;

	bool viewHistory,viewMemory;

	//Const parameters
	const int numberOfActions = 3;

	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE= 0;
};

#endif /* SRC_BOT_H_ */
