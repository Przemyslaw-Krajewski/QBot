/*
 * NeuralNetworkArray.cpp
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#include "NeuralNetworkArray.h"
#include <cstdio>

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

//	std::vector<int> layers = std::vector<int>{200,190,180,170,160,150,t_nActions};
//	std::vector<double> n = std::vector<double>{0.0000001,
//												0.00000032,
//												0.000001,
//												0.0000032,
//												0.00001,
//												0.000032,
//												0.0001};
//	double b = 8.2;

	std::vector<int> layers = std::vector<int>{200,t_nActions};
	std::vector<double> n = std::vector<double>{0.015,
												0.030};
	double b = 2.9;


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

	logLearnedValues(t_values,result);

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

void NeuralNetworkArray::logLearnedValues(std::vector<double> t_values, std::vector<double> t_result)
{
	#ifdef ENABLE_LOGGING
		std::cerr << "-----------------------------------------------------------------------\n";
		std::cerr << "Input values" << input.size() << " :\n";
		for(int i=0; i<t_values.size(); i++) std::cerr << t_values[i] << "  "; std::cerr << "\n";
		std::cerr << "Actual:\n";
		for(int i=0; i<t_result.size(); i++) std::cerr << t_result[i] << "  "; std::cerr << "\n";

		int imageSize = (input.size())/2 - 5;
		int xScreenSize = sqrt(imageSize*8/14);
		int yScreenSize = xScreenSize*14/8;

		for(int y=0; y<yScreenSize; y++)
		{
			for(int x=0; x<xScreenSize; x++)
			{
				bool block = input[(x)*yScreenSize+y]==1;
				bool enemy = input[(x)*yScreenSize+y + imageSize]==1;
				if(block && enemy) std::cerr << "W";
				else if(block) std::cerr << "#";
				else if(enemy) std::cerr << "O";
				else std::cerr << ".";
			}
			std::cerr << "\n";
		}
//		std::cerr << "\n";
		std::cerr << "\n-----------------------------------------------------------------------\n";
	#endif
}
