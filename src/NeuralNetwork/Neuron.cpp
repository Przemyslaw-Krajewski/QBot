/*
 * Komorka.cpp
 *
 *  Created on: 17 gru 2017
 *      Author: przemo
 */

#include "Neuron.h"

/*
 *
 */
Neuron::Neuron()
{
	b = nullptr;
	n = nullptr;
	output = 0;
	delta = 0;
	input.clear();
	weights.clear();
}

/*
 *
 */
Neuron::Neuron(std::vector<Neuron*> t_x, double *t_n, double *t_b) : Neuron()
{
	n = t_n;
	b = t_b;

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
Neuron::~Neuron()
{

}

/*
 *
 */
double Neuron::determineY()
{
	//sum inputs*weights
	double sum = 0;
	for(int i=0; i<input.size(); i++) sum += (input[i]->getY())*weights[i];

	//calculate result
	output = activationFunction(sum);
	delta = 0;
	return output;
}

/*
 *
 */
void Neuron::learnDeltaRule()
{
	//sum inputs*weights
	double sum = 0;
	for(int i=0; i<input.size();i++) sum+=weights[i]*(input[i]->getY());

	//determine common multiplier
	double p = (*n)*delta*derativeActivationFunction(sum);
	//calculate new weights
	for(int i=0; i<input.size(); i++)
	{
		weights[i] -= p*(input[i]->getY());
	}

	//set delta to deeper neurons
	int i = 0;
	for(std::vector<Neuron*>::iterator it=input.begin(); it!=input.end(); it++,i++)
	{
		(*it)->addToDelta(-delta*derativeActivationFunction((*it)->getSum())*weights[i]);
	}
	delta = 0;
}

/*
 *
 */
double Neuron::activationFunction(double x)
{
	return 1 / ( 1 + exp(-(*b)*x) );
}

/*
 *
 */
double Neuron::derativeActivationFunction(double x)
{
	double e = exp(-(*b)*x);
	double m = 1 + e;
	return -((*b)*e/(m*m));
}

/*
 *
 */
double Neuron::getSum()
{
	double sum = 0;
	for(int i=0; i<input.size(); i++) sum += (input[i]->getY())*weights[i];
	return sum;
}

/*
 *
 */
double Neuron::getRandomWeight()
{
	return ((double)((rand()%100))/100-0.5)*0.5;
}
