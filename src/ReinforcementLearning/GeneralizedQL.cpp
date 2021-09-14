/*
 * QLearning.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "GeneralizedQL.h"

/*
 *
 */
GeneralizedQL::GeneralizedQL(int t_nActions, int t_dimensionStatesSize) :
	qValues(HashMap(t_nActions))
{
	alpha = 0.75;
	gamma = 0.90;

	dimensionStatesSize = t_dimensionStatesSize;
	numberOfActions = t_nActions;

	controlMode = ControlMode::NN;

	actions = nullptr;
	resetActionsNN();
}

/*
 *
 */
GeneralizedQL::~GeneralizedQL()
{
	if(actions != nullptr) delete actions;
}

/*
 *
 */
void GeneralizedQL::resetActionsNN()
{
//	long size = qValues.getSize();
//	std::cout << "NN size: "<< 5+size/5 << "\n";

	if(actions != nullptr) delete actions;

	actions = new NeuralNetworkCPU::NeuralNetwork();
    actions->addLayer(new NeuralNetworkCPU::InputLayer(dimensionStatesSize));
    actions->addLayer(new NeuralNetworkCPU::SigmoidLayer(0.15,0.004, 900, actions->getLastLayerNeuronRef()));
    actions->addLayer(new NeuralNetworkCPU::SigmoidLayer(0.6,0.006, 500, actions->getLastLayerNeuronRef()));
    actions->addLayer(new NeuralNetworkCPU::SigmoidLayer(0.8,0.009, numberOfActions, actions->getLastLayerNeuronRef()));
}

/*z
 *
 */
int GeneralizedQL::chooseAction(State& t_state)
{
	std::vector<double> values;
	if(controlMode == ControlMode::NN)
	{
		values = actions->determineOutput(t_state);
	}
	else if (controlMode == ControlMode::QL)
	{
		values = qValues.getValues(t_state);
	}
	else if (controlMode == ControlMode::Hybrid || controlMode == ControlMode::NNNoLearn)
	{
		if(qValues.getChange(t_state) > ACTION_LEARN_THRESHOLD) values = actions->determineOutput(t_state);
		else values = qValues.getValues(t_state);
	}
	else assert("no such control mode" && 0);

//	for (int i=0; i< values.size() ;i++) std::cout << values[i] << "   ";
//	std::cout << "\n";

	int action = getIndexOfMaxValue(values);

	return action;
}

/*
 *
 */
double GeneralizedQL::learnSARS(SARS &t_sars)
{
	double maxValue = qValues.getValue(t_sars.state,0);
	for(int i_action=1; i_action<numberOfActions; i_action++)
	{
		double value = qValues.getValue(t_sars.state,i_action);
		if(maxValue < value) maxValue = value;
	}

	double prevValue = qValues.getValue(t_sars.oldState,t_sars.action);
	double value = prevValue + alpha*(t_sars.reward+gamma*maxValue - prevValue);
	qValues.setValue(t_sars.oldState, t_sars.action, value);

	return qValues.getChange(t_sars.oldState);
}

/*
 *
 */
std::pair<double,int> GeneralizedQL::learnAction(State &state, bool skipNotReady)
{
//	int64 timeBefore = cv::getTickCount();
	if(qValues.getChange(state) > ACTION_LEARN_THRESHOLD && skipNotReady) return std::pair<double,int>(0,2);

	std::vector<double> qlValues = qValues.getValues(state);
	std::vector<double> nnValues = actions->determineOutput(state);
	int qlAction = getIndexOfMaxValue(qlValues);
	int nnAction = getIndexOfMaxValue(nnValues);

	if(qlAction == nnAction) return std::pair<double,int>(0,1);

	std::vector<double> z;
	for(int i=0; i<numberOfActions ;i++) z.push_back(0.1);
	z[qlAction] = 0.9;

	actions->learnBackPropagation(z);

	//Calculate error;
	double err = 0;
	for(int i=0; i<nnValues.size(); i++) err += fabs(nnValues[i]-z[i]);

//	int64 timeAfter = cv::getTickCount();
//	std::cout << (timeAfter - timeBefore)/ cv::getTickFrequency() << "\n";
	return std::pair<double,int>(err, qlAction==nnAction?1:0);
}

/*
 *
 */
double GeneralizedQL::learnFromScenario(std::list<SARS> &t_history)
{
	//QLearning
	double sumErr = 0;
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = t_history.begin(); sarsIterator!=t_history.end(); sarsIterator++)
	{
		sarsIterator->reward = sarsIterator->reward + cumulatedReward;
		sumErr += abs(learnSARS(*sarsIterator));
		cumulatedReward = LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
	}

	//Learn actions
	std::vector<SARS*> shuffledStates;
	if(LEARN_FROM_HISTORY_ITERATIONS != 0)
	{
		for(std::list<SARS>::iterator sarsIterator = t_history.begin(); sarsIterator!=t_history.end(); sarsIterator++)
		{
			shuffledStates.push_back(&(*sarsIterator));
		}
	}

	for(int i=0; i<LEARN_FROM_HISTORY_ITERATIONS; i++)
	{
		std::random_shuffle(shuffledStates.begin(),shuffledStates.end());

		int skipped = 0;
		int alreadyOK = 0;
		double error = 0;
		for(std::vector<SARS*>::iterator sarsIterator = shuffledStates.begin(); sarsIterator!=shuffledStates.end(); sarsIterator++)
		{
			double learnResult = learnAction((*sarsIterator)->oldState,true).first;
			error += learnResult;
		}

		std::cout << "Error hist: " << error / ((double)t_history.size()) << "\n";
		std::cout << "Skipped hist: " << skipped << "/" << t_history.size() << "\n";
		std::cout << "Ok hist: " << alreadyOK << "/" << t_history.size()-skipped << "\n";
	}

	return sumErr;
}

/*
 *
 */
double GeneralizedQL::learnFromMemory()
{
	return 0;
//	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return;
//	if(discoveredStates.size() <= 0) return;
//
//	//Prepare states
//	std::vector<SARS*> shuffledStates;
//	for(std::map<ReducedState, SARS>::iterator i=discoveredStates.begin(); i!=discoveredStates.end(); i++) shuffledStates.push_back(&(i->second));
//
//	//Learn NN
//
//	double error = 0;
//	int skipStep = sqrt(discoveredStates.size())/10;
//	if(skipStep < 1) skipStep = 1;
//	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
//	{
//		int skipped = 0;
//		int alreadyOK = 0;
//		std::random_shuffle(shuffledStates.begin(),shuffledStates.end());
//		for(int j=0; j<shuffledStates.size(); j+=skipStep)
//		{
//			double learnResult = qLearning->learnActor((shuffledStates[j]->oldState),
//													   (shuffledStates[j]->state),
//													   (shuffledStates[j]->action),
//													   (shuffledStates[j]->reward));
//			error += learnResult;
//		}
//
//		std::cout << "Error mem: " << error / ((double) shuffledStates.size()) << "\n";
//		std::cout << "Skipped mem: " << skipped << "/" << (shuffledStates.size()/skipStep) << "\n";
//		std::cout << "Ok mem: " << alreadyOK << "/" << (shuffledStates.size()/skipStep)-skipped << "\n";
//	}
//	return error
}
