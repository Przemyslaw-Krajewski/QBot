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
#ifdef ACTORCRITICNN_LOG
	std::cout << "ACTORCRITICNN_LOG active\n";
#endif
#ifdef ACTORCRITICNN_ONE_ACTION
	std::cout << "ACTORCRITICNN_ONE_ACTION active\n";
#endif

	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

    resetNN();
}

/*
 *
 */
ActorCriticNN::~ActorCriticNN()
{
	std::remove("Memento.dat");
	std::ofstream file("Memento.dat");

	std::cout << memorizedSARS.size() << "\n";

	for(std::map<State, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++)
		LogFileHandler::printSARStoFile(file, i->second);

	file.close();
}

/*
 *
 */
void ActorCriticNN::resetNN()
{
//    actorValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
//    actorValues.addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize));
//    actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.3f,0.00008f, 2600, actorValues.getLastLayerNeuronRef()));
//    actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.6f,0.00012f, numberOfActions, actorValues.getLastLayerNeuronRef()));
//
//    criticValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
//    criticValues.addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize));
//    criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.3f, 0.00006f, 2600, criticValues.getLastLayerNeuronRef()));
//    criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.6f, 0.00009f, 1, criticValues.getLastLayerNeuronRef()));

    actorValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
    NeuralNetworkGPU::InputLayer* aIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(32,20,6));
    NeuralNetworkGPU::InputLayer* aIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-32*20*6);
    actorValues.addLayer(aIL1);
    actorValues.addLayer(aIL2);
    actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.002f,80,NeuralNetworkGPU::MatrixSize(5,5),aIL1->getNeuronPtr()));
    actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.004f,30,NeuralNetworkGPU::MatrixSize(5,5),actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.008f,10,NeuralNetworkGPU::MatrixSize(5,5),actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkGPU::FuseLayer(actorValues.getLastLayerNeuronRef(),aIL2->getNeuronPtr()));
    actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.02f,0.000012f, 1000, actorValues.getLastLayerNeuronRef()));
    actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.6f,0.000016f, numberOfActions, actorValues.getLastLayerNeuronRef()));

    criticValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
    NeuralNetworkGPU::InputLayer* cIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(32,20,6));
    NeuralNetworkGPU::InputLayer* cIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-32*20*6);
    criticValues.addLayer(cIL1);
    criticValues.addLayer(cIL2);
    criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.002f,80,NeuralNetworkGPU::MatrixSize(5,5),cIL1->getNeuronPtr()));
    criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.004f,30,NeuralNetworkGPU::MatrixSize(5,5),criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.008f,10,NeuralNetworkGPU::MatrixSize(5,5),criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new NeuralNetworkGPU::FuseLayer(criticValues.getLastLayerNeuronRef(),cIL2->getNeuronPtr()));
    criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.02f,0.000012f, 1000, criticValues.getLastLayerNeuronRef()));
    criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.6f,0.000016f, 1, criticValues.getLastLayerNeuronRef()));
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
#ifdef ACTORCRITICNN_LOG
	std::cout << critic[0] << "   :    ";
	for(int i=0; i<values.size(); i++) std::cout << values[i] << "  ";
	std::cout << "\n";
#endif
#ifdef ACTORCRITICNN_ONE_ACTION
	return 0;
#endif

	//exp
	for(int i=0; i<values.size(); i++) values[i] = exp(5*values[i]);

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
		if(randomValue < 0)
		{
#ifdef ACTORCRITICNN_LOG
			std::cout << ">>>>  " << i << "\n";
#endif
			return i;
		}
	}
#ifdef ACTORCRITICNN_LOG
	std::cout << ">>>>>>  " << values.size()-1 << "\n";
#endif
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
	std::vector<double> stateValue = criticValues.determineOutput(t_state);
	std::vector<double> criticZ = std::vector<double>();
	criticZ.push_back(t_reward);
	criticValues.learnBackPropagation(criticZ);

	//Actor
	std::vector<double> prevStateValue = criticValues.determineOutput(t_prevState);
	std::vector<double> actorZ = actorValues.determineOutput(t_prevState);

	double previousReward = UPPER_REWARD_CUP > prevStateValue[0] ? prevStateValue[0] : UPPER_REWARD_CUP;
	previousReward = LOWER_REWARD_CUP < previousReward ? previousReward : LOWER_REWARD_CUP;
	double change = t_reward-previousReward;

//	int maxValue = getIndexOfMaxValue(actorZ);
//	if(t_action == maxValue && actorZ[maxValue] > 0.90 && prevStateValue[0] > UPPER_REWARD_CUP && change > 0) return 0;

	//calculate some things
	std::vector<double> expActor;
	for(int i=0; i<actorZ.size(); i++) expActor.push_back(exp(5*actorZ[i]));
	double expSum = 0;
	for(int i=0; i<expActor.size(); i++) expSum += expActor[i];


	change = change*(1-expActor[t_action]/expSum);

	for(int i=0; i<numberOfActions; i++)
	{
		if(i!=t_action)
		{
			actorZ[i] -= change/(2);
		}
		else
		{
			actorZ[i] += change;
		}
	}


	for(int i=0 ; i<numberOfActions ; i++)
	{
		if(actorZ[i] < 0.05) actorZ[i] = 0.05;
		if(actorZ[i] > 0.95) actorZ[i] = 0.95;
	}

	actorValues.learnBackPropagation(actorZ);


	return change;
}

/*
 *
 */
double ActorCriticNN::learnFromScenario(std::list<SARS> &t_history)
{
	std::vector<SARS*> sarsPointers;
	sarsPointers.clear();

	//Temporal deference
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = t_history.begin(); sarsIterator!=t_history.end(); sarsIterator++)
	{
		cumulatedReward = sarsIterator->reward + ActorCriticNN::LAMBDA_PARAMETER*cumulatedReward;
		sarsIterator->reward = cumulatedReward;
		sarsPointers.push_back(&(*sarsIterator));

		//put state to memory
		State rs = reduceSceneState(sarsIterator->oldState,0);//sarsIterator->action
		bool exists = memorizedSARS.find(rs) != memorizedSARS.end();
		if(exists && memorizedSARS[rs].action == sarsIterator->action && sarsIterator->reward < MEMORIZE_SARS_CUP)
		{
			memorizedSARS.erase(rs);
			std::cout << "Erase state\n";
		}
		else if((exists && (memorizedSARS[rs].reward < sarsIterator->reward))
				|| (!exists && sarsIterator->reward > MEMORIZE_SARS_CUP))
		{
			memorizedSARS[rs] = SARS(sarsIterator->oldState,
									 sarsIterator->state,
									 sarsIterator->action,
									 sarsIterator->reward);
		}
	}

	double sumErr = 0;
	for(int i=0 ;i< LEARN_FROM_HISTORY_ITERATIONS ; i++ )
	{
		std::random_shuffle(sarsPointers.begin(),sarsPointers.end());

		std::vector<double> changes = std::vector<double>(numberOfActions,0);

		for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
		{
			double change = learnSARS((*sarsIterator)->oldState,
									  (*sarsIterator)->state,
									  (*sarsIterator)->action,
									  (*sarsIterator)->reward);
			changes[(*sarsIterator)->action] += change;
			cv::waitKey(15);
		}
#ifdef ACTORCRITICNN_LOG
		std::cout << "Scenario Learn Actions:\n";
		for(int i=0; i<numberOfActions; i++)
		{
			std::cout << i << ":     " << changes[i] << "\n";
		}
#endif
	}

	return sumErr/t_history.size();
}

/*
 *
 */
double ActorCriticNN::learnFromMemory()
{
#ifdef ACTORCRITICNN_ONE_ACTION
	return 0;
#endif

	int skipStep = memorizedSARS.size()/300;
	if(skipStep < 1) skipStep = 1;

#ifdef ACTORCRITICNN_LOG
	std::cout << "Memory size: " << memorizedSARS.size() << "\n";
#endif

	double sumErr = 0;
	long count = 0;
	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return 0;
	if(memorizedSARS.size() <= 0) return 0;

	//Prepare states
	std::vector<SARS*> shuffledSARS;
	for(std::map<State, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++) shuffledSARS.push_back(&(i->second));

	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
	{

		std::random_shuffle(shuffledSARS.begin(),shuffledSARS.end());

		std::vector<double> changes = std::vector<double>(numberOfActions,0);

		for(int j=0; j<shuffledSARS.size(); j+=skipStep)
		{
			double change = learnSARS((shuffledSARS[j])->oldState,
							  		  (shuffledSARS[j])->state,
									  (shuffledSARS[j])->action,
									  (shuffledSARS[j])->reward);
			changes[(shuffledSARS[j])->action] += change;
			cv::waitKey(15);
		}
#ifdef ACTORCRITICNN_LOG
		std::cout << "Memory Learn Actions:\n";
		for(int i=0; i<numberOfActions; i++)
		{
			std::cout << i << ":     " << changes[i] << "\n";
		}
#endif
	}
	return sumErr/count;
}


void ActorCriticNN::handleParameters()
{
	if(ParameterFileHandler::checkParameter("reset.param","Reset has been ordered"))
		resetNN();
}
