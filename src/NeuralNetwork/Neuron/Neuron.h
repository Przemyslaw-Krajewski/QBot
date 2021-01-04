/*
 * Komorka.h
 *
 *  Created on: 17 gru 2017
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORK_NEURON_H_
#define SRC_NEURALNETWORK_NEURON_H_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <math.h>

namespace NeuralNetworkCPU
{
	class Neuron
	{
	public:
		//basic
		virtual double getOutput() {return output;}
		virtual double determineOutput() = 0;
		virtual void learnSGD() = 0;

		//delta
		void setDelta(double nd) {delta = nd;}
		void addToDelta(double nd) {delta += nd;}
		double getDelta() {return delta;}

		virtual double getDerivative() {return 0;}

	public:
		static double getRandomWeight() { return ((double)((rand()%1000))/1000-0.5)*0.5; }

	protected:
		double delta;
		double output;
	};
}
#endif /* SRC_NEURALNETWORK_NEURON_H_ */
