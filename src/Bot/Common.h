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

const int numberOfActions = 5;
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

class ParameterFileHandler
{
public:
	static bool checkParameter(const char* t_parameterName, std::string t_communiaction)
	{
		std::ifstream hybridFile (t_parameterName);
		if (hybridFile.is_open())
		{
			hybridFile.close();
			std::remove(t_parameterName);
			std::cout << "Parameter handler: " << t_communiaction << "\n";
			return true;
		}
		return false;
	}
};

class LogFileHandler
{
public:
	static void printState(State &t_state)
	{
		int xSize = 32;
		int ySize = 56;

		std::vector<int> result;

		for(int y=0;y<56;y++)
		{
			for(int x=0;x<32;x++)
			{
				if(t_state[32*56+x*56+y] > 0) std::cout << "X";
				else if(t_state[x*56+y] > 0) std::cout << "#";
				else std::cout << ".";
			}
			std::cout << "\n";
		}
		std::cout << t_state[t_state.size()-2] << "    " << t_state[t_state.size()-1] << "\n";
		std::cout << "\n";
		std::cout << "\n";
	}

	static void printSARStoFile(std::ofstream &t_file, SARS &t_sars)
	{
		int xSize = 32;
		int ySize = 56;

		for(int y=0;y<56;y++)
		{
			for(int x=0;x<32;x++)
			{
				if(t_sars.oldState[32*56+x*56+y] > 0) t_file << "X";
				else if(t_sars.oldState[x*56+y] > 0) t_file << "#";
				else t_file << ".";
			}
			t_file << "		";
			for(int x=0;x<32;x++)
			{
				if(t_sars.state[32*56+x*56+y] > 0) t_file << "X";
				else if(t_sars.state[x*56+y] > 0) t_file << "#";
				else t_file << ".";
			}
			t_file << "\n";
		}
		int v1 = t_sars.oldState[t_sars.oldState.size()-4];
		int v2 = t_sars.oldState[t_sars.oldState.size()-3];
		int v3 = t_sars.state[t_sars.state.size()-4];
		int v4 = t_sars.state[t_sars.state.size()-3];
		t_file << v1 << "  " << v2;
		t_file << "					";
		t_file << v3 << "  " << v4 << "\n";
		t_file << t_sars.action << "  " << t_sars.reward << "\n";
		t_file << "\n";
		t_file << "\n";
	}
};
enum class ScenarioAdditionalInfo {noInfo, killedByEnemy, fallenInPitfall, notFound, timeOut, won};
enum class ControlMode {QL, NN, Hybrid, NNNoLearn};
enum class Game {BattleToads, SuperMarioBros};

#endif /* SRC_BOT_COMMON_H_ */
