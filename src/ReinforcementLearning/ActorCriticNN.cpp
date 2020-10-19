/*
 * QLearning.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "ActorCriticNN.h"

/*
 *
 */
ActorCriticNN::ActorCriticNN(int t_nActions, int t_dimensionStatesSize)
{
	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

    resetNN();
    averageReward = 0.5;
}

/*
 *
 */
void ActorCriticNN::resetNN()
{
    actorValues = NeuralNetworkGPU::NeuralNetwork();
    actorValues.addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize));
//    actorValues.addLayer(new ModifiedConvolutionalLayer(0.25, MatrixSize(3,3),25,TensorSize(32,20,6),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new PoolingLayer(MatrixSize(16,10),actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new ModifiedConvolutionalLayer(0.4, MatrixSize(5,5),60,actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new PoolingLayer(MatrixSize(8,5),actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new SigmoidLayer(0.6, 100,actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new SigmoidLayer(0.9 , numberOfActions ,actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.15,0.004, 900, actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.6,0.006, 500, actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.2,0.04, 1800, actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.3,0.06, 800, actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.7,0.09, numberOfActions, actorValues.getLastLayerNeuronRef()));

    criticValues = NeuralNetworkGPU::NeuralNetwork();
    criticValues.addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize));
    criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.2,0.04, 1800, criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.3,0.06, 800, criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.7,0.09, 1, criticValues.getLastLayerNeuronRef()));
}

/*
 *
 */
int ActorCriticNN::chooseAction(State& t_state)
{
	//getValues
	std::vector<double> values = actorValues.determineOutput(t_state);
	std::vector<double> critic = criticValues.determineOutput(t_state);

	//print
//	std::cout << critic[0] << "   :    ";
//	for(int i=0; i<values.size(); i++) std::cout << values[i] << "  ";
//	std::cout << "\n";

	//sure action
	double maxValue = getMaxValue(values);
	if(maxValue > 0.92) return getIndexOfMaxValue(values);

	//exp
	for(int i=0; i<values.size(); i++) values[i] = exp(4*values[i]);

	//Sum
	double sum = 0;
	for(int i=0; i<values.size(); i++)
	{
		sum += values[i];
	}

	//Choose random
	if(sum == 0) return rand()%numberOfActions;
	double randomValue = ((double)(rand()%((int)1000000)))/1000000;
	for(int i=0; i<values.size(); i++)
	{
		randomValue -= values[i]/sum;
		if(randomValue < 0)	return i;
	}

	return values.size()-1;
}

/*
 *
 */
double ActorCriticNN::learnSARS(State &t_prevState, State &t_state, int t_action, double t_reward)
{
	if(t_prevState.size() == 0 || t_reward == 0)
	{
		std::cout << t_prevState.size() << "  " << t_reward << "INVALID STATE!\n";
		return 0;
	}

	//Critic
	std::vector<double> prevStateValue = criticValues.determineOutput(t_prevState);
	std::vector<double> criticZ = std::vector<double>();
	criticZ.push_back(t_reward);
	criticValues.learnBackPropagation(criticZ);

	//Actor
	std::vector<double> stateValue = criticValues.determineOutput(t_state);
	std::vector<double> actorZ = actorValues.determineOutput(t_prevState);
	averageReward = averageReward*0.99 + stateValue[0]*0.01;

	//calculate some things
	std::vector<double> expActor;
	for(int i=0; i<actorZ.size(); i++) expActor.push_back(4*exp(actorZ[i]));
	double expSum = 0;
	for(int i=0; i<expActor.size(); i++) expSum += expActor[i];

	double magicReward = 0.6;
	double change = stateValue[0] < magicReward ? stateValue[0]-prevStateValue[0] : stateValue[0]-magicReward;
//	std::cout << change << "  " << stateValue[0] << "  " << prevStateValue[0] << "  " << (1-expActor[t_action]/expSum) <<  "\n";

	for(int i=0; i<numberOfActions; i++)
	{
		if(i!=t_action) actorZ[i] -= change*(1-expActor[i]/expSum);
		else
		{
			actorZ[i] += change*(1-expActor[i]/expSum);
		}
	}

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
double ActorCriticNN::learnFromScenario(std::list<SARS> &t_history)
{
	std::vector<SARS*> sarsPointers;
	sarsPointers.clear();
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = t_history.begin(); sarsIterator!=t_history.end(); sarsIterator++)
	{
		cumulatedReward = ActorCriticNN::LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
		sarsIterator->reward = cumulatedReward;
		sarsPointers.push_back(&(*sarsIterator));
		memorizedSARS[reduceSceneState(sarsIterator->oldState, sarsIterator->action)] = SARS(sarsIterator->oldState,
																							 sarsIterator->state,
																							 sarsIterator->action,
																							 sarsIterator->reward);
	}

	std::random_shuffle(sarsPointers.begin(),sarsPointers.end());

	//Learning
	double sumErr = 0;
	for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
	{
		sumErr += abs(learnSARS((*sarsIterator)->oldState,
								(*sarsIterator)->state,
								(*sarsIterator)->action,
								(*sarsIterator)->reward));
	}

	return sumErr;
}

/*
 *
 */
double ActorCriticNN::learnFromMemory()
{
	int skipStep = sqrt(memorizedSARS.size())/4;
	if(skipStep < 1) skipStep = 1;

	double sumErr = 0;
	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return 0;
	if(memorizedSARS.size() <= 0) return 0;

	//Prepare states
	std::vector<SARS*> shuffledSARS;
	for(std::map<ReducedState, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++) shuffledSARS.push_back(&(i->second));

	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
	{

		std::random_shuffle(shuffledSARS.begin(),shuffledSARS.end());
		for(int j=0; j<shuffledSARS.size(); j+=skipStep)
		{
			sumErr += abs(learnSARS((shuffledSARS[j])->oldState,
									(shuffledSARS[j])->state,
									(shuffledSARS[j])->action,
									(shuffledSARS[j])->reward));
		}
	}
	return sumErr;
}
