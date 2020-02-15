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
	InputNeuron(double t_output);
	virtual ~InputNeuron();

	virtual double determineOutput();
	virtual void learnDeltaRule();

	double setValue(double t_output);
};

#endif /* SRC_NEURALNETWORK_INPUTNEURON_H_ */