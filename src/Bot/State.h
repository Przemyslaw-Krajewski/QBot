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

struct State
{
	State();
	State(std::vector<int>& t_v);
	State(int t_size);
	State(int t_additionalInfoSize, int t_xSize, int t_ySize, int t_zSize);

	int size() {return aSize;}
	int getSizeX() {return xSize;}
	int getSizeY() {return ySize;}
	int getSizeZ() {return zSize;}
	int getImageOffset() {return imageOffset;}

	std::vector<int>& getData() {return data;}
	const std::vector<int>& getDataConst() const {return data;}

	int& operator[](int index) {return data[index];}

	std::vector<int> data;
	int aSize;

	int imageOffset;
	int xSize;
	int ySize;
	int zSize;
};
using ReducedState = State;

bool operator<(const State& state1, const State& state2);

struct SARS
{
	SARS(State t_oldState, State t_state, int t_action, double t_reward)
	{
		state = t_state;
		oldState = t_oldState;
		action = t_action;
		reward = t_reward;
		tdReward = 0;
		sumReward = 0;
		tdSumReward = t_reward;

		score = 0;

		actorValues = std::vector<double>();
		stateValue = 0;
		prevStateValue = 0;
		actorChange = 0;
	}
	SARS() : SARS(State(),State(),0,0) {}

	State state;
	State oldState;
	int action;
	double reward;
	double tdReward;
	double sumReward;
	double tdSumReward;

	int score;

	double actorChange;
	std::vector<double> actorValues;
	double stateValue;
	double prevStateValue;
};

#endif /* SRC_BOT_STATE_H_ */
