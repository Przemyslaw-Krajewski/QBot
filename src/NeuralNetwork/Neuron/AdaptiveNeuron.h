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

class AdaptiveNeuron : public Neuron
{
public:
    AdaptiveNeuron();
    AdaptiveNeuron(std::vector<Neuron*> t_x, double *t_n,
                   ActivationFunction t_af,DerivativeActivationFunction t_daf);
    virtual ~AdaptiveNeuron();

	//basic
    virtual double getOutput() {return output;}
	virtual double determineOutput();
	virtual void learnDeltaRule();

    //deriative
    void calculateDerative() { derivative = derivativeActivationFunction(sum);}
    virtual double getDerivative() {return derivative;}

    //activation function
    ActivationFunction activationFunction;
    DerivativeActivationFunction derivativeActivationFunction;

protected:
	std::vector<Neuron*> input;
    std::vector<double> weights;
    double output;
    double *n;

    double sum;
    double derivative;
};

#endif /* SRC_NEURALNETWORK_ADAPTIVENEURON_H_ */
