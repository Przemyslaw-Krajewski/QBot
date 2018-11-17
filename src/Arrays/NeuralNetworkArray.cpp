/*
 * NeuralNetworkArray.cpp
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#include "NeuralNetworkArray.h"

/*
 *
 */
NeuralNetworkArray::NeuralNetworkArray(int t_nActions, std::vector<int> t_dimensionsSize)
{
	maxValues = t_dimensionsSize;
	input = std::vector<double>(t_dimensionsSize.size());

//	std::vector<int> layers = std::vector<int>{200,190,180,170,160,150,140,t_nActions};
//	std::vector<double> n = std::vector<double>{0.000007531,
//												0.000015625,
//												0.00003125,
//												0.0000625,
//												0.000125,
//												0.00025,
//												0.0005,
//												0.001};
//	double b = 8.2;

	std::vector<int> layers = std::vector<int>{300,290,280,270,260,250,t_nActions};
	std::vector<double> n = std::vector<double>{0.000007531,
												0.000015625,
												0.00003125,
												0.0000625,
												0.000125,
												0.00025,
												0.0005};
	double b = 8.2;


	std::vector<double*> inputPtr;
	for(int i=0; i<input.size(); i++) inputPtr.push_back(&input[i]);
	neuralNetwork = new NeuralNetwork(inputPtr,layers,n,b);
}

/*
 *
 */
NeuralNetworkArray::~NeuralNetworkArray()
{
	delete neuralNetwork;
}

/*
 *
 */
double NeuralNetworkArray::getValue(std::vector<int> t_state, int t_action)
{
	setInputValues(t_state);
	std::vector<double> result = neuralNetwork->determineY();

	return result[t_action];
}

/*
 *
 */
std::vector<double> NeuralNetworkArray::getValues(std::vector<int> t_state)
{
	setInputValues(t_state);
	std::vector<double> result = neuralNetwork->determineY();

	return result;
}

/*
 *
 */
void NeuralNetworkArray::setValue(std::vector<int> t_state, int t_action, double t_value)
{
	setInputValues(t_state);
	std::vector<double> result = neuralNetwork->determineY();

	std::vector<double> z = result;
	z[t_action] = t_value;

	//learn
	neuralNetwork->learnBackPropagation(z);
}

/*
 *
 */
std::vector<double> NeuralNetworkArray::setValues(std::vector<int> t_state, std::vector<double> t_values)
{
	setInputValues(t_state);
	std::vector<double> result = neuralNetwork->determineY();

	//learn
	neuralNetwork->learnBackPropagation(t_values);

	return result;
}

/*
 *
 */
void NeuralNetworkArray::setInputValues(std::vector<int> t_state)
{
	for(int i=0; i<input.size(); i++)
	{
		input[i] = ((double) t_state[i]) / ((double) maxValues[i]);
	}
}

