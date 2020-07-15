/*
 * QLearning.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "ActorCritic.h"

/*
 *
 */
ActorCritic::ActorCritic(int t_nActions, std::vector<int> t_dimensionStatesSize)
{
	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

    resetNN();
}

/*
 *
 */
void ActorCritic::resetNN()
{
    SigmoidLayer::configure(0.14);
    actorValues = NeuralNetwork();
    actorValues.addLayer(new InputLayer(dimensionStatesSize.size()));
    actorValues.addLayer(new ModifiedConvolutionalLayer(0.25, MatrixSize(3,3),25,TensorSize(32,20,6),actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new PoolingLayer(MatrixSize(16,10),actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new ModifiedConvolutionalLayer(0.4, MatrixSize(5,5),60,actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new PoolingLayer(MatrixSize(8,5),actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new SigmoidLayer(0.6, 100,actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new SigmoidLayer(0.9 , numberOfActions ,actorValues.getLastLayerNeuronRef()));

    criticValues = NeuralNetwork();
    criticValues.addLayer(new InputLayer(dimensionStatesSize.size()));
    criticValues.addLayer(new ModifiedConvolutionalLayer(0.25, MatrixSize(3,3),15,TensorSize(32,20,6),criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new PoolingLayer(MatrixSize(16,10),criticValues.getLastLayerTensorSize(),criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new ModifiedConvolutionalLayer(0.4, MatrixSize(5,5),30,criticValues.getLastLayerTensorSize(),criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new PoolingLayer(MatrixSize(8,5),criticValues.getLastLayerTensorSize(),criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new SigmoidLayer(0.6, 100,criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new SigmoidLayer(0.9 , 1 ,criticValues.getLastLayerNeuronRef()));
}

/*
 *
 */
std::pair<bool,int> ActorCritic::chooseAction(State& t_state, ControlMode mode)
{

	std::vector<double> values = actorValues.determineOutput(t_state);
	std::vector<double> critic = criticValues.determineOutput(t_state);

	std::cout << critic[0] << "   :    ";
	double sum = 0;
	for(int i=0; i<values.size(); i++)
	{
		std::cout << values[i] << "  ";
		sum += values[i];
	}
	std::cout << "\n";
//	return std::pair<bool,int>(true,0);

	if(sum == 0) return std::pair<bool,int>(true,rand()%numberOfActions);
	double randomValue = ((double)(rand()%((int)10000)))/10000;
	for(int i=0; i<values.size(); i++)
	{
		randomValue -= values[i]/sum;
		if(values[i] > 0.90 || randomValue < 0)	return std::pair<bool,int>(true,i);
	}

	return std::pair<bool,int>(true,values.size()-1);
}

/*
 *
 */
double ActorCritic::learn(State &t_prevState, State &t_state, int t_action, double t_reward)
{
	if(t_prevState.size() == 0 || t_reward == 0)
	{
		std::cout << "INVALID STATE!\n";
		return 0;
	}

	//Critic
	std::vector<double> prevStateValue = criticValues.determineOutput(t_prevState);
	std::vector<double> criticZ = std::vector<double>();
	criticZ.push_back(t_reward);
	criticValues.learnBackPropagation(criticZ);
//	std::cout << prevStateValue[0] << "   ->   " << t_reward << "\n";

	//Actor
	std::vector<double> stateValue = criticValues.determineOutput(t_state);
	std::vector<double> actorZ = actorValues.determineOutput(t_prevState);

	double sum = 0;
	for(int i=0 ; i<numberOfActions ; i++) sum += actorZ[i];
	actorZ[t_action] = (prevStateValue[0]-stateValue[0])/actorZ[t_action] + actorZ[t_action];

	for(int i=0 ; i<numberOfActions ; i++)
	{
		if(actorZ[i] < 0.01) actorZ[i] = 0.01;
		if(actorZ[i] > 0.99) actorZ[i] = 0.99;
	}

	actorValues.learnBackPropagation(actorZ);

	return prevStateValue[0] - t_reward;
}

/*
 *
 */
NNInput ActorCritic::convertState2NNInput(const State &t_state)
{
	NNInput result;
	for(int i=0; i<t_state.size(); i++) result.push_back((double) t_state[i]);
	return result;
}

/*
 *
 */
int ActorCritic::getIndexOfMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return maxIndex;
}

/*
 *
 */
double ActorCritic::getMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return t_array[maxIndex];
}
