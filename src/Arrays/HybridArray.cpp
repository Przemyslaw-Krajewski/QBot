/*
 * NeuralNetworkArray.cpp
 *
 *  Created on: 27 lip 2018
 *      Author: przemo
 */

#include "HybridArray.h"

/*
 *
 */
HybridArray::HybridArray(int t_nActions, std::vector<int> t_dimensionsSize)
{
	acceptableError = 0.02;
	cacheSize = 200*t_nActions;

	numberOfActions = t_nActions;

	maxValues = t_dimensionsSize;
	input = std::vector<double>(t_dimensionsSize.size());

	neuralNetwork = createNeuralNetwork();

	for(int i=0; i<numberOfActions; i++) cache.push_back(CacheMemory());

//	while(1)
//	{
//		//Learn NN
//		double sumErr = 0;
//		for(int j=0; j<400; j+=numberOfActions)
//		{
//			std::vector<int> state;
//			for(int i=0; i<input.size() ;i++)
//			{
//				state.push_back(rand()%maxValues[i]);
//			}
//
//			setInputValues(state);
//			std::vector<double> z = neuralNetwork->determineY();
//			for(int i=0; i<input.size(); i++)
//			{
//				double err = fabs(z[i] - 1);
//				z[i] = 1;
//				sumErr += err;
//			}
//			neuralNetwork->learnBackPropagation(z);
//		}
//		std::cout << sumErr/400 << "\n";
//		if(sumErr/400 < acceptableError) break;
//	}
}

/*
 *
 */
HybridArray::~HybridArray()
{
	delete neuralNetwork;
}

/*
 *
 */
double HybridArray::getValue(std::vector<int> t_state, int t_action)
{
	double result;
	if(cache[t_action].count (t_state) > 0)
	{
		result = cache[t_action].find(t_state)->second;
	}
	else
	{
		setInputValues(t_state);
		std::vector<double> nnResult = neuralNetwork->determineY();
		result = nnResult[t_action];
	}
	if(result < 0.0001)
	{
		cache[t_action][t_state] = 0.99;
		result = 0.99;
	}

	return result;
}

/*
 *
 */
std::vector<double> HybridArray::getValues(std::vector<int> t_state)
{
	std::vector<double> result;
	for(int a=0; a<input.size();a++)
	{
		result.push_back(getValue(t_state,a));
	}

	return result;
}

/*
 *
 */
void HybridArray::setValue(std::vector<int> t_state, int t_action, double t_value)
{
	double value = getValue(t_state,t_action);

//	if(fabs(value-t_value) < acceptableError/4) return;

	(cache[t_action])[t_state] = t_value;

	//Erase and rewrite data if exceeded
	int sum = 0;
	for(int a=0; a<numberOfActions; a++) sum += cache[a].size();

	if(sum > cacheSize)
	{
		rewriteData();
		for(int a=0; a<numberOfActions; a++) cache[a].clear();
	}
}

/*
 *
 */
void HybridArray::setInputValues(std::vector<int> t_state)
{
	for(int i=0; i<input.size(); i++)
	{
		input[i] = ((double) t_state[i]) / ((double) maxValues[i]);
	}
}

/*
 *
 */

NeuralNetwork* HybridArray::createNeuralNetwork()
{
	std::vector<int> layers = std::vector<int>{300,260,240,220,200,180,160,numberOfActions};
	std::vector<double> n = std::vector<double>{0.000007531,
												0.000015625,
												0.00003125,
												0.0000625,
												0.000125,
												0.00025,
												0.0005,
												0.001};
	double b = 8.2;
	std::vector<double*> inputPtr;
	for(int i=0; i<input.size(); i++) inputPtr.push_back(&input[i]);
	return new NeuralNetwork(inputPtr,layers,n,b);
}

/*
 *
 */
void HybridArray::rewriteData()
{
	NeuralNetwork* newNN = createNeuralNetwork();

	int size = input.size();
	int stateDomainSize = 1000*numberOfActions;

	//Learn NN
	while(1)
	{
		double sumErr = 0;
		for(int j=0; j<stateDomainSize; j+=numberOfActions)
		{
			std::vector<int> state;
			for(int i=0; i<size ;i++)
			{
				state.push_back(rand()%maxValues[i]);
			}

			std::vector<double> values = getValues(state);
			setInputValues(state);
			std::vector<double> z = newNN->determineY();
			for(int i=0; i<size; i++)
			{
				double err = fabs(z[i] - values[i]);
				z[i] = values[i];
				sumErr += err;
			}
			if(j%100==0) std::cout <<"#";
			newNN->learnBackPropagation(z);
		}
		std::cout << sumErr/stateDomainSize << "\n";
//		newNN->modifyLearningRate(sumErr/stateDomainSize);
		if((sumErr/stateDomainSize) < acceptableError) break;
	}
	std::cout << "\n";
	delete neuralNetwork;
	neuralNetwork = newNN;
}

