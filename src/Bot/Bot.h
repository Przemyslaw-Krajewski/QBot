/*
 * Bot.h
 *
 *  Created on: 13 lut 2019
 *      Author: mistrz
 */

#ifndef SRC_BOT_H_
#define SRC_BOT_H_

#include <vector>
#include <list>
#include <math.h>

#include "Common.h"
#include "../Analyzers/StateAnalyzer.h"
#include "../QLearning/QLearning.h"
#include "../Flags.h"

class Bot {

public:

	using ScenarioResult = StateAnalyzer::AnalyzeResult::AdditionalInfo;

public:
	Bot();
	virtual ~Bot();

	void run();
	void testStateAnalyzer();

private:
	void prepareGameBeforeRun();
	bool manageScenarioTime(bool resetTimer);
	void learnQLearningScenario();

	void printScenarioResult();

	std::vector<bool> determineControllerInput(int t_action);

	std::vector<int> reduceStateResolution(std::vector<int> t_state);

	std::vector<int> copySceneState(cv::Mat& image, std::vector<bool>& controllerInput, StateAnalyzer::Point& position, StateAnalyzer::Point& velocity);
	std::vector<int> createSceneState(cv::Mat& sceneState, std::vector<bool>& controllerInput, StateAnalyzer::Point& position, StateAnalyzer::Point& velocity);
	StateAnalyzer::AnalyzeResult extractSceneState(std::vector<int> sceneData);


private:
	//
	StateAnalyzer analyzer;
	QLearning *qLearning;

	//Scenario data
	std::vector<int> sceneState;
	std::vector<bool> controllerInput;
	int action = 0;
	int time;
	int randomAction;

	//Scenario result data
	std::list<SARS> historyScenario;
	ScenarioResult scenarioResult;

	//Learning data
	int persistCounter;
	VisitedSARS visitedSARS;


	//Const parameters
	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE= 0;
	const int TIME_LIMIT = 90;
	const int ITERATIONS_BEFORE_NN_PERSIST{1};
	const int LEARNING_LOOPS{1};
};

#endif /* SRC_BOT_H_ */
