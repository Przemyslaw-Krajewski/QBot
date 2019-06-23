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

class Neuron {
public:
	Neuron();
	Neuron(std::vector<Neuron*> t_x, double *t_n, double *t_b);
	virtual ~Neuron();

	//basic
	virtual double determineY();
	virtual double getY() {return output;}
	virtual void learnDeltaRule();
	//delta
	void setDelta(double nd) {delta = nd;}
	void addToDelta(double nd) {delta += nd;}
	double getDelta() {return delta;}
	//additional
	std::vector<double> getW() const {return weights;}
	void setW(std::vector<double> t_w) {weights=t_w;}
	double getSum();
	//activation function
	double activationFunction(double x);
	double derativeActivationFunction(double x);

private:
	double getRandomWeight();

protected:
public:
	std::vector<Neuron*> input;
	std::vector<double> weights;
	double output;
	double delta;
	double *n;
	double *b;

	double sum;
};

#endif /* SRC_NEURALNETWORK_NEURON_H_ */
