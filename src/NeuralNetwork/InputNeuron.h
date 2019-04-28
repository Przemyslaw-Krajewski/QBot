/*
 * KomorkaWejsciowa.h
 *
 *  Created on: 21 gru 2017
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORK_INPUTNEURON_H_
#define SRC_NEURALNETWORK_INPUTNEURON_H_

#include "Neuron.h"

class InputNeuron : public Neuron
{
public:
	InputNeuron();
	InputNeuron(double t_x);
	virtual ~InputNeuron();

	virtual double determineY();
	virtual double getY();
	virtual void learnBackPropagation();

	double setY(double t_y) {x = t_y;}

private:
	double x;
};

#endif /* SRC_NEURALNETWORK_INPUTNEURON_H_ */
