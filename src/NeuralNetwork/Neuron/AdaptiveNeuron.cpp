/*
 * Komorka.cpp
 *
 *  Created on: 17 gru 2017
 *      Author: przemo
 */

#include "AdaptiveNeuron.h"

/*
 *
 */
AdaptiveNeuron::AdaptiveNeuron()
{
	n = nullptr;
	output = 0;
	delta = 0;
	input.clear();
	weights.clear();

	sum = 0;
    derivative = 0;
}

/*
 *
 */
AdaptiveNeuron::AdaptiveNeuron(std::vector<Neuron*> t_x, double *t_n,
                               ActivationFunction t_af,DerivativeActivationFunction t_daf) : AdaptiveNeuron()
{
	n = t_n;
    activationFunction = t_af;
    derivativeActivationFunction = t_daf;

	//create neurons
	for(int i=0; i<t_x.size(); i++)
	{
		weights.push_back(getRandomWeight());
		input.push_back(t_x[i]);
	}
}

/*
 *
 */
AdaptiveNeuron::~AdaptiveNeuron()
{

}

/*
 *
 */
double AdaptiveNeuron::determineOutput()
{
	//sum inputs*weights
	sum = 0;
	for(int i=0; i<input.size(); i++) sum += (input[i]->getOutput()) * weights[i];

	//calculate result
	output = activationFunction(sum);
	delta = 0;
	return output;
}

/*
 *
 */
void AdaptiveNeuron::learnDeltaRule()
{
	//determine common multiplier
	calculateDerative();
	double p = (*n) * delta * derivative;
	//calculate new weights
	for(int i=0; i<input.size(); i++)
	{
		weights[i] -= p*(input[i]->getOutput());
	}

	//set delta to deeper neurons
	int i = 0;
	for(std::vector<Neuron*>::iterator it=input.begin(); it!=input.end(); it++,i++)
	{
		(*it)->addToDelta(-delta * derivative * weights[i]);
	}
	delta = 0;
}