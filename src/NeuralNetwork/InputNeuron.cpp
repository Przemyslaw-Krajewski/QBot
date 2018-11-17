/*
 * KomorkaWejsciowa.cpp
 *
 *  Created on: 21 gru 2017
 *      Author: przemo
 */

#include "InputNeuron.h"

InputNeuron::InputNeuron(double* t_x) : Neuron()
{
	x = t_x;
}

InputNeuron::~InputNeuron()
{

}

double InputNeuron::determineY()
{
	output = *x;
	delta = 0;
	return output;
}

double InputNeuron::getY()
{
	output = *x;
	return output;
}

void InputNeuron::learnBackPropagation()
{

}
