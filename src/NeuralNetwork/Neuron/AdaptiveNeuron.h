/*
 * Komorka.h
 *
 *  Created on: 17 gru 2017
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORK_ADAPTIVENEURON_H_
#define SRC_NEURALNETWORK_ADAPTIVENEURON_H_

#include "Neuron.h"

using ActivationFunction = double (*)(double);
using DerivativeActivationFunction = double (*)(double);

namespace NeuralNetworkCPU
{
	class AdaptiveNeuron : public Neuron
	{

	private:
		AdaptiveNeuron();
	public:
		AdaptiveNeuron(std::vector<Neuron*> t_x, double *t_n,
					   ActivationFunction t_af,DerivativeActivationFunction t_daf);
		AdaptiveNeuron(std::vector<Neuron*> t_x, double *t_n, std::vector<double> * t_weights,
					   ActivationFunction t_af,DerivativeActivationFunction t_daf);
		AdaptiveNeuron(const AdaptiveNeuron& t_an);
		virtual ~AdaptiveNeuron();

		//basic
		virtual double getOutput() {return output;}
		virtual double determineOutput();
		virtual void learnDeltaRule();

		//cumulative
		void cumulativeLearnDeltaRule();
		void cumulationReset() {for (int i=0; i<cumulativeChange.size();i++) cumulativeChange[i] = 0;}
		void cumulationApply() {for (int i=0; i<cumulativeChange.size();i++) (*weights)[i] += cumulativeChange[i];}

		//activation function
		ActivationFunction activationFunction;
		DerivativeActivationFunction derivativeActivationFunction;

		//getters
		std::vector<Neuron*> getInput() const {return input;}
		std::vector<double>* getWeights() const {return weights;}
		bool getCommonWeights() const {return commonWeights;}
		double* getLearnRate() const {return n;}
		ActivationFunction getActivationFunction() const {return activationFunction;}
		DerivativeActivationFunction getDerivativeActivationFunction() const {return derivativeActivationFunction;}

	protected:
		std::vector<Neuron*> input;
		std::vector<double> *weights;
		std::vector<double> cumulativeChange;
		bool commonWeights;

		double output;
		double *n;

		double sum;
		double derivative;
	};
}
#endif /* SRC_NEURALNETWORK_ADAPTIVENEURON_H_ */
