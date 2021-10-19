/*
 * Common.h
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_BOT_STATE_H_
#define SRC_BOT_STATE_H_

#include <opencv2/opencv.hpp>
#include <vector>

const int numberOfActions = 3;

using State = std::vector<int>;
using ReducedState = State;

struct Point
{
	Point() {x=y=0;}
	Point(int t_x, int t_y) {x=t_x;y=t_y;}
	Point& operator=(const cv::Point& t_p ) {x=t_p.x;y=t_p.y;return *this;}
	Point& operator=(const Point & t_p ) {x=t_p.x;y=t_p.y;return *this;}
	bool operator==(const Point & t_p ) {return (x==t_p.x && y==t_p.y);}
	int x;
	int y;
};

struct StateInfo
{
	StateInfo() : StateInfo(-1,-1,-1) {}
	StateInfo(int t_x, int t_y, int t_z) {xSize=t_x;ySize=t_y;zSize=t_z;}

	int xSize;
	int ySize;
	int zSize;
};

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
