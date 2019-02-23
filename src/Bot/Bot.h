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

#include "../Analyzers/StateAnalyzer.h"
#include "../QLearning/QLearning.h"
#include "../Flags.h"

class Bot {
	using DeathReason = StateAnalyzer::AnalyzeResult::AdditionalInfo;

	struct HistoryEntry
	{
		HistoryEntry(State t_oldState, State t_state, int t_action, double t_reward)
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

	void run();

private:
	void prepareGameBeforeRun();
	bool manageScenarioTime(bool resetTimer);
	void learnQLearningScenario();

	void printDeathReason();

	std::vector<bool> determineControllerInput(int t_action);

	std::vector<int> copySceneState(cv::Mat& image, std::vector<bool>& controllerInput, StateAnalyzer::Point& position, StateAnalyzer::Point& velocity);
	std::vector<int> createSceneState(cv::Mat& sceneState, std::vector<bool>& controllerInput, StateAnalyzer::Point& position, StateAnalyzer::Point& velocity);
private:
	//
	StateAnalyzer analyzer;
	QLearning *qLearning;

	//Scenario data
	std::vector<int> sceneState;
	std::vector<bool> controllerInput;
	int action = 0;
	std::list<HistoryEntry> historyScenario;
	DeathReason deathReason;
	int time;

	//Const parameters
	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE= 0;
	const int TIME_LIMIT = 150;
};

#endif /* SRC_BOT_H_ */
