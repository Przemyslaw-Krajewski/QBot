/*
 * Komorka.cpp
 *
 *  Created on: 17 gru 2017
 *      Author: przemo
 */

#include "AdaptiveNeuron.h"

namespace NeuralNetworkCPU
{
	/*
	 *
	 */
	AdaptiveNeuron::AdaptiveNeuron()
	{
		n = nullptr;
		output = 0;
		delta = 0;
		input.clear();
		weights = nullptr;
		commonWeights = true;

		sum = 0;
		derivative = 0;

		activationFunction = nullptr;
		derivativeActivationFunction = nullptr;
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

		weights = new std::vector<double>;
		commonWeights = false;
		//create neurons
		for(int i=0; i<t_x.size(); i++)
		{
			weights->push_back(getRandomWeight());
			cumulativeChange.push_back(0);
			input.push_back(t_x[i]);
		}
	}

	/*
	 *
	 */
	AdaptiveNeuron::AdaptiveNeuron(std::vector<Neuron*> t_x, double *t_n, std::vector<double>* t_weights,
								   ActivationFunction t_af,DerivativeActivationFunction t_daf) : AdaptiveNeuron()
	{
		n = t_n;
		activationFunction = t_af;
		derivativeActivationFunction = t_daf;

		weights = t_weights;
		commonWeights = true;
		//create neurons
		for(int i=0; i<t_x.size(); i++)
		{
			cumulativeChange.push_back(0);
			input.push_back(t_x[i]);
		}
	}

	/*
	 *
	 */
	AdaptiveNeuron::AdaptiveNeuron(const AdaptiveNeuron& t_an)
	{
		output = 0;
		delta = 0;
		sum = 0;
		derivative = 0;

		n = t_an.getLearnRate();
		input = t_an.getInput();

		activationFunction = t_an.getActivationFunction();
		derivativeActivationFunction = t_an.getDerivativeActivationFunction();

		commonWeights = t_an.getCommonWeights();
		if(!commonWeights)
		{
			weights = new std::vector<double>();
			for(int i=0; i< t_an.getWeights()->size(); i++) weights->push_back(getRandomWeight());
		}
		else  weights = t_an.getWeights();
	}

	/*
	 *
	 */
	AdaptiveNeuron::~AdaptiveNeuron()
	{
		if(!commonWeights) delete weights;
	}

	/*
	 *
	 */
	double AdaptiveNeuron::determineOutput()
	{
		//sum inputs*weights
		sum = 0;
		for(int i=0; i<input.size(); i++)
		{
			sum += (input[i]->getOutput()) * (*weights)[i];
		}

		//calculate result
		output = activationFunction(sum);
		delta = 0;
		return output;
	}

	/*
	 *
	 */
	void AdaptiveNeuron::learnSGD()
	{
		//determine common multiplier
		derivative = derivativeActivationFunction(sum);
		double p = (*n) * delta * derivative;
		//calculate new weights
		for(int i=0; i<input.size(); i++)
		{
			(*weights)[i] -= p*(input[i]->getOutput());
		}

		//set delta to deeper neurons
		int i = 0;
		for(std::vector<Neuron*>::iterator it=input.begin(); it!=input.end(); it++,i++)
		{
			(*it)->addToDelta(delta * derivative * (*weights)[i]);
		}
		delta = 0;
	}

	/*
	 *
	 */
	void AdaptiveNeuron::cumulativeLearnDeltaRule()
	{
		//determine common multiplier
		derivative = derivativeActivationFunction(sum);
		double p = (*n) * delta * derivative;
		//calculate new weights
		for(int i=0; i<input.size(); i++)
		{
			cumulativeChange[i] -= p*(input[i]->getOutput());
		}

		//set delta to deeper neurons
		int i = 0;
		for(std::vector<Neuron*>::iterator it=input.begin(); it!=input.end(); it++,i++)
		{
			(*it)->addToDelta(delta * derivative * (*weights)[i]);
		}
		delta = 0;
	}
}
