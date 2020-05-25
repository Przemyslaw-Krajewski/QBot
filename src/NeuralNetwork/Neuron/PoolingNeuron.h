/*
 * Komorka.h
 *
 *  Created on: 17 gru 2017
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORK_POOLINGNEURON_H_
#define SRC_NEURALNETWORK_POOLINGNEURON_H_

#include "Neuron.h"

namespace NeuralNetworkCPU
{
	class PoolingNeuron : public Neuron
	{
	private:
		PoolingNeuron();
	public:
		PoolingNeuron(std::vector<Neuron*> t_x);
		virtual ~PoolingNeuron();

		//basic
		virtual double getOutput() {return output;}
		virtual double determineOutput();
		virtual void learnDeltaRule();

		//getters
		std::vector<Neuron*> getInput() const {return input;}

	protected:
		std::vector<Neuron*> input;

		double output;
	};
}
#endif /* SRC_NEURALNETWORK_POOLINGNEURON_H_ */
