/*
 * ReinforcementLearning.h
 *
 *  Created on: 16 wrz 2020
 *      Author: przemo
 */

#ifndef SRC_REINFORCEMENTLEARNING_REINFORCEMENTLEARNING_H_
#define SRC_REINFORCEMENTLEARNING_REINFORCEMENTLEARNING_H_

#include <vector>
#include <list>
#include <random>

#include "../Bot/State.h"

struct LearningStatus
{
	LearningStatus(double t_e, char t_ui) {learnError = t_e; userInput = t_ui;}

	char userInput;
	double learnError;
};

class ReinforcementLearning
{

public:
	ReinforcementLearning();
	virtual ~ReinforcementLearning();

	//Basic methods
	virtual int chooseAction(State& t_state) = 0;
	virtual double learnSARS(SARS &t_sars) = 0;

	virtual LearningStatus learnFromScenario(std::list<SARS> &t_history) = 0;
	virtual LearningStatus learnFromMemory() = 0;

	virtual void handleUserInput(char t_userInput) {};


protected:
//	NNInput convertState2NNInput(const State &t_state);
	int getIndexOfMaxValue(std::vector<double> t_array);
	double getMaxValue(std::vector<double> t_array);
	int getWeightedRandom(std::vector<double> t_array);
};

#endif /* SRC_REINFORCEMENTLEARNING_REINFORCEMENTLEARNING_H_ */
