/*
 * ReinforcementLearning.h
 *
 *  Created on: 16 wrz 2020
 *      Author: przemo
 */

#ifndef SRC_REINFORCEMENTLEARNING_REINFORCEMENTLEARNING_H_
#define SRC_REINFORCEMENTLEARNING_REINFORCEMENTLEARNING_H_

#include <vector>

#include "../Bot/Common.h"

class ReinforcementLearning
{
public:
	ReinforcementLearning();
	virtual ~ReinforcementLearning();

	//Basic methods
	virtual int chooseAction(State& t_state) = 0;
	virtual double learnSARS(State &t_prevState, State &t_state, int t_action, double t_reward) = 0;

	virtual double learnFromScenario(std::list<SARS> &t_history) = 0;
	virtual double learnFromMemory() = 0;


protected:
//	NNInput convertState2NNInput(const State &t_state);
	int getIndexOfMaxValue(std::vector<double> t_array);
	double getMaxValue(std::vector<double> t_array);
	State reduceSceneState(const State& t_state, double action);
};

#endif /* SRC_REINFORCEMENTLEARNING_REINFORCEMENTLEARNING_H_ */
