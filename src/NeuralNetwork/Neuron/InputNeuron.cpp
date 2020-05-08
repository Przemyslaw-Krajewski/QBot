/*
 * KomorkaWejsciowa.cpp
 *
 *  Created on: 21 gru 2017
 *      Author: przemo
 */

#include "InputNeuron.h"

namespace CPUNeuralNetwork
{
	InputNeuron::InputNeuron()
	{
		output = 0;
		delta = 0;
	}

	InputNeuron::InputNeuron(double t_output) : InputNeuron()
	{
		output = t_output;
	}

	InputNeuron::~InputNeuron()
	{
		//Do nothing
	}

	double InputNeuron::determineOutput()
	{
		delta = 0;
		return output;
	}

	void InputNeuron::learnDeltaRule()
	{
		//Do nothing
	}

	double InputNeuron::setValue(double t_output)
	{
		output = t_output;
		return output;
	}
}
