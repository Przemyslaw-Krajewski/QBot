/*
 * Common.h
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_BOT_COMMON_H_
#define SRC_BOT_COMMON_H_

#include <opencv2/opencv.hpp>

#include <vector>

const int numberOfActions = 8;
const int numberOfControllerInputs = 6;

using State = std::vector<int>;
using ReducedState = State;
using ControllerInput = std::vector<bool>;

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

enum class ScenarioAdditionalInfo {noInfo, killedByEnemy, fallenInPitfall, notFound, timeOut, won};
enum class ControlMode {QL, NN, Hybrid, NNNoLearn};
enum class Game {BattleToads, SuperMarioBros};

#endif /* SRC_BOT_COMMON_H_ */
