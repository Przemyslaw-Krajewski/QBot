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

#include "Common.h"
#include "../Analyzers/StateAnalyzer.h"
#include "../QLearning/QLearning.h"
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
	void learnFromScenario(std::list<SARS> &historyScenario);
	void learnFromMemory();
	void eraseInvalidLastStates(std::list<SARS> &history);

	ControllerInput determineControllerInput(int t_action);
	State createSceneState(cv::Mat& sceneState, ControllerInput& controllerInput, Point& position, Point& velocity);
	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extractSceneState(std::vector<int> sceneData);

	static State reduceStateResolution(const State& t_state)
	{
		int reduceLevel = 2;
		std::vector<int> result;
		for(int i=0;i<t_state.size();i++)
		{
			if(i%reduceLevel!=0 ||( ((int)i/56)%reduceLevel!=0) ) continue;
			result.push_back(t_state[i]);
		}

		return result;
	}

private:
	//
	StateAnalyzer analyzer;
	QLearning *qLearning;

	std::map<ReducedState,State> discoveredStates;

	//Const parameters
	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE= 0;
	const int TIME_LIMIT = 80;
	const int LEARN_FROM_HISTORY_ITERATIONS =1;
	const int LEARN_FROM_MEMORY_ITERATIONS = 4;
};

#endif /* SRC_BOT_H_ */
