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
	DesktopHandler::getPtr()->releaseControllerButton();
	DesktopHandler::loadGame();

	//Initialize scene data
	StateAnalyzer::AnalyzeResult analyzeResult;

	analyzeResult = analyzer.analyzeBT();
	ControllerInput controllerInput = determineControllerInput(0);
	State sceneState = createSceneState(analyzeResult.processedImage,
								  analyzeResult.processedImagePast,
								  analyzeResult.processedImagePast2,
								  controllerInput,
								  analyzeResult.playerCoords,
								  analyzeResult.playerVelocity);
	DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
	cv::waitKey(1000);

	//Initialize qLearning
	actorCritic = new ActorCritic(numberOfActions, std::vector<int>(sceneState.size(),255));

	playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
}

/*
 *
 */
Bot::~Bot()
{
	delete actorCritic;
}

/*
 *
 */
void Bot::execute()
{
	double bestScore = -999999;
	double averageScore = 0;

	while(1)
	{
		//State variables
		std::list<SARS> historyScenario;
		State sceneState;
		ControllerInput controllerInput = determineControllerInput(0);
		int action = 0;
		ScenarioResult scenarioResult = ScenarioResult::noInfo;
//		int time = TIME_LIMIT;

		//Reload game
		DesktopHandler::getPtr()->releaseControllerButton();
		DesktopHandler::loadGame();
		cv::waitKey(3000);

		//Get first state
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyzeBT();
		sceneState = createSceneState(analyzeResult.processedImage,
						analyzeResult.processedImagePast,
						analyzeResult.processedImagePast2,
						controllerInput,
						analyzeResult.playerCoords,
						analyzeResult.playerVelocity);

		//Info variables
		long discoveredStatesSize = discoveredStates.size();
		while(1)
		{
			//Persist info
#ifdef PRINT_PROCESSING_TIME
			int64 timeBefore = cv::getTickCount();
#endif

			std::vector<int> oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyzeBT();
//			if(analyzeResult.fieldAndEnemiesLayout.cols == 0 || analyzeResult.fieldAndEnemiesLayout.rows == 0) continue;
			sceneState = createSceneState(analyzeResult.processedImage,
										  analyzeResult.processedImagePast,
										  analyzeResult.processedImagePast2,
						 	 	 	 	  controllerInput,
										  analyzeResult.playerCoords,
										  analyzeResult.playerVelocity);

			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));
//			discoveredStates[reduceStateResolution(sceneState)] = sceneState;

			//Determine new controller input
			action = actorCritic->chooseAction(sceneState, controlMode).second;
			controllerInput = determineControllerInput(action);
			DesktopHandler::getPtr()->pressControllerButton(controllerInput);

			//Draw info
//			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extraxtedSceneData = extractSceneState(sceneState);
//			DataDrawer::drawAnalyzedData(extraxtedSceneData.first,extraxtedSceneData.second,
//					analyzeResult.reward,qLearning->getQChange(sceneState));

#ifdef PRINT_PROCESSING_TIME
			int64 afterBefore = cv::getTickCount();
			std::cout << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
#endif
			//End?
			if(analyzeResult.endScenario)
			{
				scenarioResult = analyzeResult.additionalInfo;
				break;
			}
		}
		DesktopHandler::getPtr()->releaseControllerButton();

		std::cout << "\n";

		loadParameters();

		//Reset?
		if(reset)
		{
			std::cout << "RESET NN\n";
			actorCritic->resetActionsNN();
			reset = false;
		}

//		std::cout << "Added: " << discoveredStates.size() - discoveredStatesSize << "  Discovered: " << discoveredStates.size() << "\n";
		learnFromScenarioAC(historyScenario);

		//Learning
//		learnFromMemory();
//		learnFromScenario(historyScenario);
//		playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
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
void Bot::learnFromScenarioAC(std::list<SARS> &historyScenario)
{

	std::vector<SARS*> sarsPointers;
	sarsPointers.clear();
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
	{
		cumulatedReward = ActorCritic::LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
		sarsIterator->reward += cumulatedReward;
		sarsPointers.push_back(&(*sarsIterator));
	}

	std::random_shuffle(sarsPointers.begin(),sarsPointers.end());

	//QLearning
	double sumErr = 0;
	for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
	{
		sumErr += abs(actorCritic->learn((*sarsIterator)->oldState,
										 (*sarsIterator)->state,
										 (*sarsIterator)->action,
										 (*sarsIterator)->reward));
	}

	std::cout << "Cumulated reward: " << cumulatedReward << "\n";
	std::cout << "History size: " << historyScenario.size() << " sumErr: " << sumErr << "\n";
	std::cout << "Sum Error: " << sumErr/historyScenario.size() << "\n";

}

/*
 *
 */
void Bot::learnFromScenario(std::list<SARS> &historyScenario)
{

//	if(LEARN_FROM_HISTORY_ITERATIONS == 0) return;
//
//	for(int i=0; i<LEARN_FROM_HISTORY_ITERATIONS; i++)
//	{
//		int skipped = 0;
//		int alreadyOK = 0;
//		double error = 0;
//		for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
//		{
//			if(sarsIterator->action == -1) continue;
//
//			std::pair<double,int> learnResult = qLearning->learnAction(&sarsIterator->oldState, true);
//
//			if( learnResult.second == 2) skipped++; // QL not sure
//			else if(learnResult.second == 3 || learnResult.second == 1) alreadyOK++;
//			else error += learnResult.first;
//
////			if(learnResult.second == 2 || learnResult.second == 3) sarsIterator->action = -1;
//		}
//
//		std::cout << "Error hist: " << error / ((double)historyScenario.size()-skipped-alreadyOK) << "\n";
//		std::cout << "Skipped hist: " << skipped << "/" << alreadyOK << "/" << historyScenario.size() << "\n";
//	}
}

/*
 *
 */
void Bot::learnFromMemory()
{
//	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return;
//	if(discoveredStates.size() <= 0) return;
//
//	//Prepare states
//	std::vector<const State*> shuffledStates;
//	for(std::map<ReducedState, State>::iterator i=discoveredStates.begin(); i!=discoveredStates.end(); i++) shuffledStates.push_back(&(i->second));
//
//	//Learn NN
//	int skipStep = 1;
//	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
//	{
//		int skipped = 0;
//		int alreadyOK = 0;
//		double error = 0;
//		std::random_shuffle(shuffledStates.begin(),shuffledStates.end());
//		for(int j=0; j<shuffledStates.size(); j+=skipStep)
//		{
//
//			std::pair<double,int> learnResult = qLearning->learnAction(&(*(shuffledStates[j])));
//
//			if( learnResult.second == 2) skipped++; // QL not sure
//			else if(learnResult.second == 3 || learnResult.second == 1) alreadyOK++;
//			else error += learnResult.first;
//		}
//
//		std::cout << "Error mem: " << error / ((double) shuffledStates.size()) << "\n";
//		std::cout << "Skipped mem: "<< skipped << "/" << alreadyOK << "/" << (shuffledStates.size()/skipStep) << "\n" ;
//	}
}

/*
 *
 */
void Bot::eraseNotReadyStates()
{
//	long erased = 0;
//	for(std::map<ReducedState, State>::iterator it = discoveredStates.begin(); it!=discoveredStates.end();)
//	{
//		double change = qLearning->getQChange(it->second);
//		if(change > QLearning::ACTION_LEARN_THRESHOLD)
//		{
//			erased++;
//			it = discoveredStates.erase(it);
//		}
//		else it++;
//	}
//
//	std::cout << "Erased: " << erased << " states  =>  " << discoveredStates.size() << "\n";
}

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
State Bot::reduceStateResolution(const State& t_state)
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

	return result;
}

/*
 *
 */
std::vector<int> Bot::createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
		ControllerInput& controllerInput, Point& position, Point& velocity)
{
	State sceneState;

	for(int x=0; x<image.cols; x++)
	{
		for(int y=0; y<image.rows; y++)
		{
			uchar* ptrSrc = image.ptr(y)+(3*(x));
			sceneState.push_back((ptrSrc[0] >> 6) + (ptrSrc[1] >> 4) + (ptrSrc[2] >> 2));
		}
	}

	for(int x=0; x<imagePast.cols; x++)
	{
		for(int y=0; y<imagePast.rows; y++)
		{
			uchar* ptrSrc = imagePast.ptr(y)+(3*(x));
			sceneState.push_back((ptrSrc[0] >> 7) + (ptrSrc[1] >> 6) + (ptrSrc[2] >> 5));
		}
	}

	for(int x=0; x<imagePast2.cols; x++)
	{
		for(int y=0; y<imagePast2.rows; y++)
		{
			uchar* ptrSrc = imagePast2.ptr(y)+(3*(x));
			sceneState.push_back((ptrSrc[0] >> 7) + (ptrSrc[1] >> 6) + (ptrSrc[2] >> 5));
		}
	}

	//Controller
	for(bool ci : controllerInput)
	{
		if(ci==true) sceneState.push_back(MAX_INPUT_VALUE);
		else sceneState.push_back(MIN_INPUT_VALUE);
	}
	return sceneState;
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
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyzeBT();
		//Print info
//		DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
		std::cout << ": " << analyzeResult.reward << "\n";
	}
}
