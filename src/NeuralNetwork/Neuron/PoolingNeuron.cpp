/*
 * Komorka.cpp
 *
 *  Created on: 17 gru 2017
 *      Author: przemo
 */

#include "PoolingNeuron.h"

namespace NeuralNetworkCPU
{
	/*
	 *
	 */
	PoolingNeuron::PoolingNeuron()
	{
		output = 0;
		delta = 0;
		input.clear();
	}

	/*
	 *
	 */
	PoolingNeuron::PoolingNeuron(std::vector<Neuron*> t_x) : PoolingNeuron()
	{
		//create neurons
		for(int i=0; i<t_x.size(); i++)
		{
			input.push_back(t_x[i]);
		}
	}

	/*
	 *
	 */
	PoolingNeuron::~PoolingNeuron()
	{

	}

	/*
	 *
	 */
	double PoolingNeuron::determineOutput()
	{
		//sum inputs*weights
		double sum = 0;
		for(int i=0; i<input.size(); i++) sum += (input[i]->getOutput());

		//calculate result
		output = sum/input.size();
		return output;
	}

	/*
	 *
	 */
	void PoolingNeuron::learnDeltaRule()
	{
		//set delta to deeper neurons
		int i = 0;
		for(std::vector<Neuron*>::iterator it=input.begin(); it!=input.end(); it++,i++)
		{
			(*it)->addToDelta(-delta);
		}
		delta = 0;
	}
}
