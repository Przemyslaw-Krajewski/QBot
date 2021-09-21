/*
 * Common.h
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_BOT_STATE_H_
#define SRC_BOT_STATE_H_

#include <vector>

const int numberOfActions = 2;

using State = std::vector<int>;
using ReducedState = State;

struct SARS
{
	SARS()
	{
		state = State();
		oldState = State();
		reward = 0;
		action = 0;
	}
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


enum class ScenarioAdditionalInfo {ok, killedByEnemy, killedByEnvironment, playerNotFound, timeOut, won};
enum class Game {BattleToads, SuperMarioBros};

#endif /* SRC_BOT_STATE_H_ */
