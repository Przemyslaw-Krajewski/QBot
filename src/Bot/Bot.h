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
#include "../Loggers/DataDrawer.h"

class Bot {
	using ScenarioResult = StateAnalyzer::AnalyzeResult::AdditionalInfo;
	using ReducedState = State;

	struct SARS
	{
		SARS(State t_oldState, State t_state, int t_action, double t_reward)
		{
			state = t_state;
			oldState = t_oldState;
			reward = t_reward;
			action = t_action;
		}

		State state;
		State oldState;
		int action;
		double reward;
	};

public:
	Bot();
	virtual ~Bot();

	void execute();
	void testStateAnalyzer();

private:
	void loadParameters();

	void learnFromScenarioAC(std::list<SARS> &historyScenario);
	void learnFromScenario(std::list<SARS> &historyScenario);
	void learnFromMemory();
	void eraseNotReadyStates();

	ControllerInput determineControllerInput(int t_action);
	State createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
						   ControllerInput& controllerInput, Point& position, Point& velocity);
	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extractSceneState(std::vector<int> sceneData);

	static State reduceStateResolution(const State& t_state);

private:
	//
	StateAnalyzer analyzer;
	ActorCritic *actorCritic;

	std::map<ReducedState, State> discoveredStates;
	int playsBeforeNNLearning;

	ControlMode controlMode;
	bool reset;

	//Const parameters
	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE= 0;
	const int TIME_LIMIT = 80;
	const int LEARN_FROM_HISTORY_ITERATIONS = 2;
	const int LEARN_FROM_MEMORY_ITERATIONS  = 3;
	const int PLAYS_BEFORE_NEURAL_NETWORK_LEARNING = 15;
};

#endif /* SRC_BOT_H_ */
