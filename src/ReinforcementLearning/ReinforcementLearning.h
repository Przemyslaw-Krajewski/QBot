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
#include <algorithm>

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

	virtual void prepareScenario(std::list<SARS> &t_history, int additionalInfo) {}
	virtual LearningStatus learnFromScenario(std::list<SARS> &t_history) = 0;
	virtual LearningStatus learnFromMemory() = 0;

	virtual void handleUserInput(char t_userInput) {}
	virtual std::map<State, SARS>* getMemoryMap() {return nullptr;}


protected:
	int getWeightedRandom(std::vector<double> t_array);

private:
    std::random_device randomDevice;
    std::mt19937 randomNumberGenerator;
};

#endif /* SRC_REINFORCEMENTLEARNING_REINFORCEMENTLEARNING_H_ */
