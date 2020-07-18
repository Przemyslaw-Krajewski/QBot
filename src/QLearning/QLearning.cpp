/*
 * QLearning.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "QLearning.h"

/*
 *
 */
QLearning::QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize)
{
	alpha = 1;
	gamma = 0.975;

	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

	qValues = nullptr;
	target = nullptr;
	actor = nullptr;
	resetActionsNN();
}

/*
 *
 */
QLearning::~QLearning()
{
	if(qValues != nullptr) delete qValues;
	if(target != nullptr) delete target;
	if(actor != nullptr) delete actor;
}

/*
 *
 */
void QLearning::resetActionsNN()
{
//	long size = qValues.getSize();
//	std::cout << "NN size: "<< 5+size/5 << "\n";

	if(qValues != nullptr) delete qValues;
	qValues = new NeuralNetworkGPU::NeuralNetwork();
	qValues->addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize.size()));
//	qValues->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.15,0.004, 900, qValues->getLastLayerNeuronRef()));
	qValues->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.4,0.04, 900, qValues->getLastLayerNeuronRef()));
	qValues->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.6,0.06, numberOfActions, qValues->getLastLayerNeuronRef()));

	if(target != nullptr) delete target;
	target = new NeuralNetworkGPU::NeuralNetwork();
	target->addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize.size()));
//	target->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.15,0.004, 900, target->getLastLayerNeuronRef()));
	target->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.4,0.04, 900, target->getLastLayerNeuronRef()));
	target->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.6,0.06, numberOfActions, target->getLastLayerNeuronRef()));
	copyQValuesToTarget();

	if(actor != nullptr) delete actor;
	actor= new NeuralNetworkGPU::NeuralNetwork();
	actor->addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize.size()));
//	actor->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.15,0.004, 900, actor->getLastLayerNeuronRef()));
	actor->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.4,0.04, 900, actor->getLastLayerNeuronRef()));
	actor->addLayer(new NeuralNetworkGPU::SigmoidLayer(0.6,0.06, numberOfActions, actor->getLastLayerNeuronRef()));
}

/*
 *
 */
void QLearning::copyQValuesToTarget()
{
	auto tIt = target->layers.begin();
	auto qvIt = qValues->layers.begin();
	while(qvIt!=qValues->layers.end() || tIt!=target->layers.end())
	{
		(*tIt)->writeWeights((*qvIt)->readWeights());
		qvIt++;tIt++;
	}
	std::cout << "Copy QValues -> Target\n";
}

/*
 *
 */
std::pair<bool,int> QLearning::chooseAction(State& t_state, ControlMode mode)
{
	std::vector<double> values = qValues->determineOutput(t_state);

//	for (int i=0; i< values.size() ;i++) std::cout << values[i] << "   ";
//	std::cout << "\n";

	int action = getIndexOfMaxValue(values);

	return std::pair<bool,int>(true,action);
}

/*
 *
 */
std::pair<bool,int> QLearning::chooseActorAction(State& t_state, ControlMode mode)
{
//	return std::pair<bool,int>(true,0);
	std::vector<double> values = actor->determineOutput(t_state);
//	std::vector<double> critic = qValues->determineOutput(t_state);

	double sum = 0;
	for(int i=0; i<values.size(); i++)
	{
//		std::cout << values[i] << "  ";
		sum += values[i];
	}
//	std::cout << "\n";

//	std::cout << "      ";
//	for(int i=0; i<critic.size(); i++)
//	{
//		std::cout << critic[i] << "  ";
//	}
//	std::cout << "\n";

	if(sum == 0) return std::pair<bool,int>(true,rand()%numberOfActions);
	double randomValue = ((double)(rand()%((int)10000)))/10000;
	for(int i=0; i<values.size(); i++)
	{
		randomValue -= values[i]/sum;
		if(values[i] > 0.80 || randomValue < 0)	return std::pair<bool,int>(true,i);
	}

	return std::pair<bool,int>(true,values.size()-1);
}


/*
 *
 */
double QLearning::learnQL(State t_prevState, State t_state, int t_action, double t_reward)
{
	std::vector<double> targetStateValues = qValues->determineOutput(t_state);
	std::vector<double> prevStateValues = qValues->determineOutput(t_prevState);

	//QValues
	double maxValue = targetStateValues[t_action];
	//	double maxValue = targetStateValues[0];
//	for(int i_action=1; i_action<numberOfActions; i_action++)
//	{
//		double value = targetStateValues[i_action];
//		if(maxValue < value) maxValue = value;
//	}

	double prevStateValue = prevStateValues[t_action];
	double value = prevStateValue + alpha*(t_reward+gamma*maxValue - prevStateValue);

	std::vector<double> qValuesZ = prevStateValues;

	double error = qValuesZ[t_action] - value;
	qValuesZ[t_action] = value > 0.9 ? 0.9 : value;

	if(t_reward < 0.001) for(int i=0; i< qValuesZ.size();i++) qValuesZ[i] = t_reward;

//	std::cout << prevStateValues[t_action] << "  ->  " << qValuesZ[t_action] << "  " << maxValue << "  " << t_reward << "\n";
	qValues->learnBackPropagation(qValuesZ);

	//Actor
	std::vector<double> actorZ = actor->determineOutput(t_prevState);

	double sum = 0;
	for(int i=0 ; i<numberOfActions ; i++) sum += actorZ[i];
	actorZ[t_action] = -(prevStateValues[t_action]-targetStateValues[t_action])/actorZ[t_action] + actorZ[t_action];

	for(int i=0 ; i<numberOfActions ; i++)
	{
		if(actorZ[i] < 0.1) actorZ[i] = 0.1;
		if(actorZ[i] > 0.9) actorZ[i] = 0.9;
	}

	actor->learnBackPropagation(actorZ);

	return error;
}

/*
 *
 */
NeuralNetworkGPU::NNInput QLearning::convertState2NNInput(const State &t_state)
{
	NeuralNetworkGPU::NNInput result;
	for(int i=0; i<t_state.size(); i++) result.push_back((double) t_state[i]);
	return result;
}

/*
 *
 */
int QLearning::getIndexOfMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return maxIndex;
}
