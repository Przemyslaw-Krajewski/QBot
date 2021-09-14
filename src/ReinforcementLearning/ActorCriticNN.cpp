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
ActorCriticNN::ActorCriticNN(int t_nActions, int t_dimensionStatesSize, StateAnalyzer *t_stateAnalyzer)
{
#ifdef ACTORCRITICNN_LOG
	std::cout << "ACTORCRITICNN_LOG active\n";
#endif
#ifdef ACTORCRITICNN_ONE_ACTION
	std::cout << "ACTORCRITICNN_ONE_ACTION active\n";
#endif

	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;
	stateAnalyzer = t_stateAnalyzer;

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
    createNNA();
}

void ActorCriticNN::createNN0()
{
	actorValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	actorValues.addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.3f,0.00008f, 3600, actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.00012f, numberOfActions, actorValues.getLastLayerNeuronRef()));

	criticValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	criticValues.addLayer(new NeuralNetworkGPU::InputLayer(dimensionStatesSize));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.3f, 0.00006f, 3600, criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f, 0.00009f, 1, criticValues.getLastLayerNeuronRef()));

}

void ActorCriticNN::createNNA()
{
	actorValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* aIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* aIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	actorValues.addLayer(aIL2);
	actorValues.addLayer(aIL1);
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000009,120,NeuralNetworkGPU::MatrixSize(3,3),aIL1->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::PoolingLayer(actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015f,6,NeuralNetworkGPU::MatrixSize(3,3),actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::FuseLayer(actorValues.getLastLayerNeuronRef(),aIL2->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000009f, 3500, actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000015f, numberOfActions, actorValues.getLastLayerNeuronRef()));

	criticValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* cIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* cIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	criticValues.addLayer(cIL2);
	criticValues.addLayer(cIL1);
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000009f,100,NeuralNetworkGPU::MatrixSize(3,3),cIL1->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::PoolingLayer(criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015f,6,NeuralNetworkGPU::MatrixSize(3,3),criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::FuseLayer(criticValues.getLastLayerNeuronRef(),cIL2->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000009f, 3500, criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000015f, 1, criticValues.getLastLayerNeuronRef()));
}

void ActorCriticNN::createNNB()
{
	actorValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* aIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* aIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	actorValues.addLayer(aIL2);
	actorValues.addLayer(aIL1);
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015f,120,NeuralNetworkGPU::MatrixSize(3,3),aIL1->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::PoolingLayer(actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000025f,6,NeuralNetworkGPU::MatrixSize(3,3),actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::FuseLayer(actorValues.getLastLayerNeuronRef(),aIL2->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000015f, 3500, actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000025f, numberOfActions, actorValues.getLastLayerNeuronRef()));

	criticValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* cIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* cIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	criticValues.addLayer(cIL2);
	criticValues.addLayer(cIL1);
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015f,100,NeuralNetworkGPU::MatrixSize(3,3),cIL1->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::PoolingLayer(criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000025f,6,NeuralNetworkGPU::MatrixSize(3,3),criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::FuseLayer(criticValues.getLastLayerNeuronRef(),cIL2->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000015f, 3500, criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000025f, 1, criticValues.getLastLayerNeuronRef()));
}

void ActorCriticNN::createNNC()
{
	actorValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* aIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* aIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	actorValues.addLayer(aIL2);
	actorValues.addLayer(aIL1);
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015,100,NeuralNetworkGPU::MatrixSize(3,3),aIL1->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::PoolingLayer(actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000025f,6,NeuralNetworkGPU::MatrixSize(3,3),actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000009f, 2500, actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::FuseLayer(actorValues.getLastLayerNeuronRef(),aIL2->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000015f, 1500, actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.8f,0.000025f, numberOfActions, actorValues.getLastLayerNeuronRef()));

	criticValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* cIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* cIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	criticValues.addLayer(cIL2);
	criticValues.addLayer(cIL1);
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015f,80,NeuralNetworkGPU::MatrixSize(3,3),cIL1->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::PoolingLayer(criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000025f,6,NeuralNetworkGPU::MatrixSize(3,3),criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000009f, 2500, criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::FuseLayer(criticValues.getLastLayerNeuronRef(),cIL2->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000015f, 1500, criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.8f,0.000025f, 1, criticValues.getLastLayerNeuronRef()));
}

void ActorCriticNN::createNND()
{
	actorValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* aIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* aIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	actorValues.addLayer(aIL2);
	actorValues.addLayer(aIL1);
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015,120,NeuralNetworkGPU::MatrixSize(3,3),aIL1->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::PoolingLayer(actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000025f,6,NeuralNetworkGPU::MatrixSize(3,3),actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::FuseLayer(actorValues.getLastLayerNeuronRef(),aIL2->getNeuronPtr()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000015f, 3500, actorValues.getLastLayerNeuronRef()));
	actorValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000025f, numberOfActions, actorValues.getLastLayerNeuronRef()));

	criticValues = NeuralNetworkGPU::NeuralNetwork(NeuralNetworkGPU::LearnMode::Adam);
	NeuralNetworkGPU::InputLayer* cIL1 = new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(64,40,6));
	NeuralNetworkGPU::InputLayer* cIL2 = new NeuralNetworkGPU::InputLayer(dimensionStatesSize-64*40*6);
	criticValues.addLayer(cIL2);
	criticValues.addLayer(cIL1);
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000015f,100,NeuralNetworkGPU::MatrixSize(3,3),cIL1->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::PoolingLayer(criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.01f,0.000025f,6,NeuralNetworkGPU::MatrixSize(3,3),criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::FuseLayer(criticValues.getLastLayerNeuronRef(),cIL2->getNeuronPtr()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.012f,0.000015f, 3500, criticValues.getLastLayerNeuronRef()));
	criticValues.addLayer(new NeuralNetworkGPU::SigmoidLayer(0.5f,0.000025f, 1, criticValues.getLastLayerNeuronRef()));
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
	std::cout << floor(critic[0]*100000)/100000 << "\t:\t";
	for(int i=0; i<values.size(); i++) std::cout << floor(values[i]*100000)/100000 << "\t";
	std::cout << "\n";
#endif
#ifdef ACTORCRITICNN_ONE_ACTION
	return 1;
#endif

	//exp
	int m = 1;
	if(critic[0] > 0.8) m = 9;
	else if(critic[0] > 0.7) m = 5;
	else m = 2;
	for(int i=0; i<values.size(); i++) values[i] = exp(m*values[i]);
	int action = getWeightedRandom(values);
#ifdef ACTORCRITICNN_LOG
	std::cout << ">>>>  " << action << "\n";
#endif
	return action;
}

/*
 *
 */
double ActorCriticNN::learnSARS(SARS &t_sars)
{
	if(t_sars.oldState.size() == 0 || t_sars.reward == 0)
	{
		std::cout << t_sars.oldState.size() << "  " << t_sars.reward << "INVALID STATE!\n";
		return 0;
	}

	//Critic
	std::vector<double> stateValue = criticValues.determineOutput(t_sars.state);
	std::vector<double> criticZ = std::vector<double>();
	criticZ.push_back(t_sars.reward);
	criticValues.setMeanSquareDelta(criticZ);
	criticValues.learnBackPropagation();

	//Actor
	std::vector<double> prevStateValue = criticValues.determineOutput(t_sars.oldState);
	std::vector<double> actorZ = actorValues.determineOutput(t_sars.oldState);

	double previousReward = UPPER_REWARD_CUP > prevStateValue[0] ? prevStateValue[0] : UPPER_REWARD_CUP;
	previousReward = LOWER_REWARD_CUP < previousReward ? previousReward : LOWER_REWARD_CUP;
	double change = t_sars.reward-previousReward;

	actorValues.setSoftMaxDelta(actorZ, change, t_sars.action);
	actorValues.learnBackPropagation();

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

		putStateToMemory(*sarsIterator);
	}

	//Learning
	for(int i=0 ;i< LEARN_FROM_HISTORY_ITERATIONS ; i++ )
	{
		processLearningFromSARS(sarsPointers);
	}

	return 0;
}

/*
 *
 */
double ActorCriticNN::learnFromMemory()
{
#ifdef ACTORCRITICNN_ONE_ACTION
	return 0;
#endif
#ifdef ACTORCRITICNN_LOG
	std::cout << "Memory size: " << memorizedSARS.size() << "\n";
#endif

	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return 0;
	if(memorizedSARS.size() <= 0) return 0;

	//Prepare states
	std::vector<SARS*> shuffledSARS;
	for(std::map<State, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++) shuffledSARS.push_back(&(i->second));

	//Learning
	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
	{
		processLearningFromSARS(shuffledSARS);
	}
	return 0;
}


void ActorCriticNN::handleParameters()
{
	if(ParameterFileHandler::checkParameter("reset.param","ActorCritic::Reset has been ordered"))
	{
		resetNN();
	}
	else if(ParameterFileHandler::checkParameter("save.param","ActorCritic::Save NN to file"))
	{
		criticValues.saveToFile("CriticNN.dat");
		actorValues.saveToFile("ActorNN.dat");
	}
	else if(ParameterFileHandler::checkParameter("load.param","ActorCritic::Load NN from file"))
	{
		criticValues.loadFromFile("CriticNN.dat");
		actorValues.loadFromFile("ActorNN.dat");
	}
}

void ActorCriticNN::processLearningFromSARS(std::vector<SARS*> t_sars)
{
	std::random_shuffle(t_sars.begin(),t_sars.end());

	std::vector<double> changesp = std::vector<double>(numberOfActions,0);
	std::vector<double> changesm = std::vector<double>(numberOfActions,0);

	for(std::vector<SARS*>::iterator sarsIterator = t_sars.begin(); sarsIterator!=t_sars.end(); sarsIterator++)
	{
		double change = learnSARS(*(*sarsIterator));

		if(change >0) changesp[(*sarsIterator)->action] += change;
		else 		  changesm[(*sarsIterator)->action] += change;

		cv::waitKey(40);
	}
#ifdef ACTORCRITICNN_LOG
	std::cout << "Scenario Learn Actions:\n";
	for(int i=0; i<numberOfActions; i++)
	{
		std::cout << i << ":     " << changesp[i] << "  " << changesm[i] << "\n";
	}
#endif
}

void ActorCriticNN::putStateToMemory(SARS &t_sars)
{
	State rs = stateAnalyzer->reduceSceneState(t_sars.oldState,t_sars.action);
	bool exists = memorizedSARS.find(rs) != memorizedSARS.end();
	if(exists && memorizedSARS[rs].action == t_sars.action && t_sars.reward < MEMORIZE_SARS_CUP)
	{
		if(memorizedSARS[rs].state == t_sars.state && memorizedSARS[rs].oldState == t_sars.oldState)
		{
			std::cout << "Erase state:" << t_sars.action << "  " << memorizedSARS[rs].action << "  " << t_sars.reward << "\n";
			memorizedSARS.erase(rs);
		}
	}
	else if((exists && (memorizedSARS[rs].reward < t_sars.reward))
			|| (!exists && t_sars.reward > MEMORIZE_SARS_CUP))
	{
		memorizedSARS[rs] = t_sars;
	}
}
