/*
 * Bot.cpp
 *
 *  Created on: 13 lut 2019
 *      Author: mistrz
 */

#include "Bot.h"

/*
 *
 */
Bot::Bot()
{
	reset = false;
	controlMode = ControlMode::QL;

	/*  Initialize */
	cv::waitKey(3000);
	//Load game in order to avoid not finding player during initializing
	MemoryAnalyzer::getPtr()->setController(0);
	MemoryAnalyzer::getPtr()->loadState();

	//Initialize scene data
	StateAnalyzer::AnalyzeResult analyzeResult;
	for(int i=1; i<11; i++)
	{
		analyzeResult = stateAnalyzer.analyze();
		if(analyzeResult.additionalInfo != ScenarioAdditionalInfo::notFound) break;
		cv::waitKey(1000);
		std::cout << "Could not find player, atteption: " << i << "\n";
	}
	if(analyzeResult.additionalInfo == ScenarioAdditionalInfo::notFound)
				throw std::string("Could not initialize, check player visibility");

	ControllerInput controllerInput = determineControllerInput(0);
	State sceneState = stateAnalyzer.createSceneState(analyzeResult.processedImage,
													  analyzeResult.processedImagePast,
													  analyzeResult.processedImagePast2,
													  controllerInput,
													  analyzeResult.playerCoords,
													  analyzeResult.playerVelocity);
//	DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
	cv::waitKey(1000);

	//Initialize acLearning
	reinforcementLearning = new QLearning(numberOfActions, (int) sceneState.size());

	playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
}

/*
 *
 */
Bot::~Bot()
{
	delete reinforcementLearning;
}

/*
 *
 */
void Bot::execute()
{
	while(1)
	{
		double score = 0;

		//State variables
		std::list<SARS> historyScenario;
		State sceneState;
		ControllerInput controllerInput = determineControllerInput(0);
		int action = 0;
		ScenarioAdditionalInfo scenarioResult = ScenarioAdditionalInfo::noInfo;
		int time = TIME_LIMIT;


		//Reload game
		MemoryAnalyzer::getPtr()->setController(0);
		MemoryAnalyzer::getPtr()->loadState();
		cv::waitKey(100);

		//Get first scene state
		StateAnalyzer::AnalyzeResult analyzeResult;
		for(int i=1; i<11; i++)
		{
			analyzeResult = stateAnalyzer.analyze();
			if(analyzeResult.additionalInfo != ScenarioAdditionalInfo::notFound) break;
			cv::waitKey(1000);
			std::cout << "Could not find player, atteption: " << i << "\n";
		}
		if(analyzeResult.additionalInfo == ScenarioAdditionalInfo::notFound)
					throw std::string("Could not initialize, check player visibility");

		sceneState = stateAnalyzer.createSceneState(analyzeResult.processedImage,
													analyzeResult.processedImagePast,
													analyzeResult.processedImagePast2,
													controllerInput,
													analyzeResult.playerCoords,
													analyzeResult.playerVelocity);

		while(1)
		{
			//Persist info
#ifdef PRINT_PROCESSING_TIME
			int64 timeBefore = cv::getTickCount();
#endif
			cv::waitKey(40);
			std::vector<int> oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = stateAnalyzer.analyze();
			sceneState = stateAnalyzer.createSceneState(analyzeResult.processedImage,
														analyzeResult.processedImagePast,
														analyzeResult.processedImagePast2,
														controllerInput,
														analyzeResult.playerCoords,
														analyzeResult.playerVelocity);
			if(analyzeResult.reward >= StateAnalyzer::LITTLE_ADVANCE_REWARD ) score++ ;
			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));

			//Determine new controller input
			action = reinforcementLearning->chooseAction(sceneState);
			controllerInput = determineControllerInput(action);
			MemoryAnalyzer::getPtr()->setController(determineControllerInputInt(action));

			//Draw info
//			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extraxtedSceneData = extractSceneState(sceneState);
//			DataDrawer::drawAnalyzedData(extraxtedSceneData.first,extraxtedSceneData.second,
//					analyzeResult.reward,actorCritic->getQChange(sceneState));

#ifdef PRINT_PROCESSING_TIME
			int64 afterBefore = cv::getTickCount();
			std::cout << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
#endif
			//Timer
			if(analyzeResult.reward < StateAnalyzer::LITTLE_ADVANCE_REWARD) time--;
			else if(time < TIME_LIMIT) time++;

			//End?
			if(analyzeResult.endScenario || time<0)
			{
				scenarioResult = analyzeResult.additionalInfo;
				break;
			}
		}

		MemoryAnalyzer::getPtr()->setController(0);

		loadParameters();

//		std::cout << score << "\n";

		stateAnalyzer.correctScenarioHistory(historyScenario, scenarioResult);

		reinforcementLearning->learnFromScenario(historyScenario);
		reinforcementLearning->learnFromMemory();
	}
}

/*
 *
 */
void Bot::loadParameters()
{
	std::ifstream qlFile ("ql.param");
	if (qlFile.is_open())
	{
		qlFile.close();
		std::remove("ql.param");
		controlMode = ControlMode::QL;
		std::cout << "QL mode activated\n";
	}

	std::ifstream nnFile ("nn.param");
	if (nnFile.is_open())
	{
		nnFile.close();
		std::remove("nn.param");
		controlMode = ControlMode::NN;
		std::cout << "NN mode activated\n";
	}

	std::ifstream hybridFile ("hybrid.param");
	if (hybridFile.is_open())
	{
		hybridFile.close();
		std::remove("hybrid.param");
		controlMode = ControlMode::Hybrid;
		std::cout << "Hybrid mode activated\n";
		playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
	}

	std::ifstream nnNoLearnFile ("nnnolearn.param");
	if (nnNoLearnFile.is_open())
	{
		nnNoLearnFile.close();
		std::remove("nnnolearn.param");
		controlMode = ControlMode::NNNoLearn;
		std::cout << "NN no learn mode activated\n";
	}

	std::ifstream resetFile ("reset.param");
	if (resetFile.is_open())
	{
		resetFile.close();
		std::remove("reset.param");
		reset = true;
		std::cout << "Reset has been ordered\n";
	}
}

/*
 *
 */
//void Bot::learnFromScenarioAC(std::list<SARS> &historyScenario)
//{
//	std::vector<SARS*> sarsPointers;
//	sarsPointers.clear();
//	double cumulatedReward = 0;
//	for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
//	{
//		cumulatedReward = ActorCritic::LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
//		sarsIterator->reward = cumulatedReward;
//		sarsPointers.push_back(&(*sarsIterator));
//		memorizedSARS[reduceSceneState(sarsIterator->oldState, sarsIterator->action)] = SARS(sarsIterator->oldState,
//																								  sarsIterator->state,
//																								  sarsIterator->action,
//																								  sarsIterator->reward);
//		double value = actorCritic->getCriticValue((sarsIterator)->oldState);
////		std::cout << value << " = "<< (sarsIterator)->reward << "\n";
//	}
//
//	long counter=0;
//	std::random_shuffle(sarsPointers.begin(),sarsPointers.end());
//
//	//Learning
//	double sumErr = 0;
//	for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
//	{
//		sumErr += abs(actorCritic->learn((*sarsIterator)->oldState,
//										 (*sarsIterator)->state,
//										 (*sarsIterator)->action,
//										 (*sarsIterator)->reward));
//	}
////	std::cout << sumErr/sarsPointers.size() << "\n";
////	actorCritic->drawCriticValues();
//
////	while(0)
////	{
////
////		std::random_shuffle(sarsPointers.begin(),sarsPointers.end());
////
////		//Learning
////		double sumErr = 0;
////		for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
////		{
////			sumErr += abs(actorCritic->learn((*sarsIterator)->oldState,
////											 (*sarsIterator)->state,
////											 (*sarsIterator)->action,
////											 (*sarsIterator)->reward));
////		}
////		std::cout << sumErr/sarsPointers.size() << "\n";
////
////		sumErr = 0;
////		counter++;
////		for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
////		{
////			double value = actorCritic->getCriticValue((*sarsIterator)->oldState);
////			sumErr += abs(value-(*sarsIterator)->reward);
////			//std::cout << value << " = "<< (*sarsIterator)->reward << "\n";
////		}
////		std::cout << sumErr/sarsPointers.size() << "  " << sarsPointers.size() << "  " << counter << "\n";
////		if(counter >200) { int p=0;p=3/p;}
//////		if(counter%10==0) actorCritic->drawCriticValues();
////	}
//}

/*
 *
 */
//void Bot::learnFromMemoryAC()
//{
//	int skipStep = memorizedSARS.size()/150;
//	if(skipStep < 1) skipStep = 1;
//
//	double sumErr = 0;
//	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return;
//	if(memorizedSARS.size() <= 0) return;
//
//	//Prepare states
//	std::vector<SARS*> shuffledSARS;
//	for(std::map<ReducedState, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++) shuffledSARS.push_back(&(i->second));
//
//	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
//	{
//
//		std::random_shuffle(shuffledSARS.begin(),shuffledSARS.end());
//		for(int j=0; j<shuffledSARS.size(); j+=skipStep)
//		{
//			sumErr += abs(actorCritic->learn((shuffledSARS[j])->oldState,
//										     (shuffledSARS[j])->state,
//										     (shuffledSARS[j])->action,
//										     (shuffledSARS[j])->reward));
//		}
//	}
//}

/*
 *
 */
ControllerInput Bot::determineControllerInput(int t_action)
{
	ControllerInput w;
	for(int i=0; i<numberOfControllerInputs; i++) w.push_back(false);

	w[ 2+(t_action%4) ] = true;
	w[0] = t_action>3;

	return w;
}

/*
 *
 */
int Bot::determineControllerInputInt(int t_action)
{
//	int direction = (1<<(4+t_action%4));
//	int jump = t_action>3?1:0;
//	return direction+jump;
	switch(t_action)
	{
	case 0: //Right
		return 128;
	case 1: //Right jump
		return 128+1;
	case 2: //Left
		return 64;
	case 3: //Jump
		return 1;
	case 4: //Left Jump
		return 64+1;
	default:
		std::cout << t_action << "\n";
		assert("No such action!" && false);
	}
	return 0;
}

/*
 *
 */
State Bot::reduceSceneState(const State& t_state, double action)
{
	int reduceLevel = 4;
	std::vector<int> result;
	for(int i=0;i<t_state.size()-10;i++)
	{
		if(i%reduceLevel!=0) continue;
		result.push_back(t_state[i]);
	}
	for(int i=t_state.size()-10;i<t_state.size();i++)
	{
		result.push_back(t_state[i]);
	}
	result.push_back(action);

	return result;
}

/*
 *
 */
std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> Bot::extractSceneState(State sceneState)
{
	cv::Mat fieldAndEnemiesLayout = cv::Mat(56, 32, CV_8UC3);

	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			ptr[0]=0;
			ptr[1]=0;
			ptr[2]=0;
		}
	}

	//Terrain
	long i=0;
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(sceneState[i] == 1) {ptr[0]=100; ptr[1]=100; ptr[2]=100;}
			else {ptr[0]=0; ptr[1]=0; ptr[2]=0;}
			i++;
		}
	}
	//Enemies
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(sceneState[i] == 1) {ptr[0]=0; ptr[1]=0; ptr[2]=220;}
			i++;
		}
	}

	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> result ;
//	result.first.fieldAndEnemiesLayout = fieldAndEnemiesLayout;
	//TODO disabled

//	//AdditionalInfo
//	result.first.playerCoords.x = sceneState[sceneState.size()-4];
//	result.first.playerCoords.y = sceneState[sceneState.size()-3];
//	result.first.playerVelocity.x = sceneState[sceneState.size()-2];
//	result.first.playerVelocity.y = sceneState[sceneState.size()-1];

	//Controller
	for(int i=0;i<6; i++) result.second.push_back(sceneState[sceneState.size()-1+i]);

	return result;
}

/*
 *
 */
void Bot::testStateAnalyzer()
{
	while(1)
	{
		StateAnalyzer::AnalyzeResult analyzeResult = stateAnalyzer.analyze();
		//Print info
//		DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
		std::cout << ": " << analyzeResult.reward << "\n";
	}
}
