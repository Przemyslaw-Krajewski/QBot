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
}

/*
 *
 */
void ActorCriticNN::resetNN()
{
    actorValues = NeuralNetworkCPU::NeuralNetwork();
    actorValues.addLayer(new NeuralNetworkCPU::InputLayer(dimensionStatesSize));
//    actorValues.addLayer(new ModifiedConvolutionalLayer(0.25, MatrixSize(3,3),25,TensorSize(32,20,6),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new PoolingLayer(MatrixSize(16,10),actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new ModifiedConvolutionalLayer(0.4, MatrixSize(5,5),60,actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new PoolingLayer(MatrixSize(8,5),actorValues.getLastLayerTensorSize(),actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new SigmoidLayer(0.6, 100,actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new SigmoidLayer(0.9 , numberOfActions ,actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.15,0.004, 900, actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.6,0.006, 500, actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.3,0.04, 900, actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.7,0.06, 500, actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(1.0,0.09, numberOfActions, actorValues.getLastLayerNeuronRef()));

    criticValues = NeuralNetworkCPU::NeuralNetwork();
    criticValues.addLayer(new NeuralNetworkCPU::InputLayer(dimensionStatesSize));
    criticValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.3,0.04, 900, criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(0.7,0.06, 500, criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new NeuralNetworkCPU::SigmoidLayer(1.0,0.09, 1, criticValues.getLastLayerNeuronRef()));
}

/*
 *
 */
int ActorCriticNN::chooseAction(State& t_state)
{
//	return 3;
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
	return 0;

	if(sum == 0) return rand()%numberOfActions;
	double randomValue = ((double)(rand()%((int)100000)))/100000;
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

//	std::cout << prevStateValue[0] << " -> " << t_reward << "\n";

//	//Actor
	std::vector<double> stateValue = criticValues.determineOutput(t_state);
	std::vector<double> actorZ = actorValues.determineOutput(t_prevState);

//	std::cout << stateValue[0] << "  " << prevStateValue[0] << "  " << actorZ[t_action] << "  " << log2(actorZ[t_action]) << "\n";
//	std::cout << -(stateValue[0]-prevStateValue[0])*log(actorZ[t_action]) << "\n";

	actorZ[t_action] = -(stateValue[0]-prevStateValue[0])*log2(actorZ[t_action]) + actorZ[t_action];//(stateValue[0]-prevStateValue[0]);



	for(int i=0 ; i<numberOfActions ; i++)
	{
		if(actorZ[i] < 0.001) actorZ[i] = 0.01;
		if(actorZ[i] > 0.999) actorZ[i] = 0.999;
	}
//	std::cout << actorZ[t_action] << "\n";
//	std::cout << "\n";

	actorValues.learnBackPropagation(actorZ);
//	double sum = 0;
//	for(int i=0; i<actorZ.size(); i++) {sum += actorZ[i];}
//	for(int i=0; i<actorZ.size(); i++) actorZ[i] = actorZ[i]/sum;

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
//		double value = getCriticValue((sarsIterator)->oldState);
//		std::cout << sarsIterator->reward << "  " << value << "\n";
	}

	long counter=0;
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
	//	std::cout << sumErr/sarsPointers.size() << "\n";
	//	actorCritic->drawCriticValues();

		while(0)
		{

//			std::random_shuffle(sarsPointers.begin(),sarsPointers.end());

			//Learning
			double sumErr = 0;
			for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
			{
				sumErr += abs(learnSARS((*sarsIterator)->oldState,
												 (*sarsIterator)->state,
												 (*sarsIterator)->action,
												 (*sarsIterator)->reward));
			}
//			std::cout << sumErr/sarsPointers.size() << "\n";

//			sumErr = 0;
			counter++;
//			for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
//			{
//				double value = criticValues.determineOutput((*sarsIterator)->oldState)[0];
//				sumErr += abs(value-(*sarsIterator)->reward);
//				//std::cout << value << " = "<< (*sarsIterator)->reward << "\n";
//			}
			std::cout << sumErr/sarsPointers.size() << "  " << sarsPointers.size() << "  " << counter << "\n";
			if(counter >200) { int p=0;p=3/p;}
	//		if(counter%10==0) actorCritic->drawCriticValues();
		}
	return sumErr;
}

/*
 *
 */
double ActorCriticNN::learnFromMemory()
{
	int skipStep = memorizedSARS.size()/150;
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
