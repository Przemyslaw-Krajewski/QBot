/*
 * ActionCritic.h
 *
 *  Created on: 23 paź 2018
 *      Author: przemo
 */

#ifndef SRC_ACTIONCRITIC_ACTIONCRITIC_H_
#define SRC_ACTIONCRITIC_ACTIONCRITIC_H_

#include <vector>
#include <string>

#include "../Game/Point.h"
#include "../Arrays/Array.h"
#include "../Arrays/Table.h"
#include "../Arrays/NeuralNetworkArray.h"
#include "../Arrays/HybridArray.h"

using State = std::vector<int>;

class ActionCritic {
public:
	ActionCritic(int t_nActions, std::vector<int> t_dimensionStatesSize);
	virtual ~ActionCritic();

public:
	double learnActor(State t_prevState, State t_state, int t_action, double t_reward);
	double learnCritic(State t_prevState, State t_state, int t_action, double t_reward);
	int chooseAction(State t_state);

	void setCriticValue(State t_state, double t_value) {criticValues->setValue(t_state, 0, t_value);}
	double getQValue(State t_state) { return criticValues->getValue(t_state,0);}

	void printCriticMap(State state, std::string string);
	void printActorMap(State state, std::string string);

private:
	double gamma;
	double alpha;
	int numberOfActions;

	Array *criticValues;
	Array *actorValues;
};

#endif /* SRC_ACTIONCRITIC_ACTIONCRITIC_H_ */
