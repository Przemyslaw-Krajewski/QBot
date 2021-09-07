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
//		std::cout << "Could not find player, atteption: " << i << "\n";
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
	cv::waitKey(1000);

	//Initialize acLearning
	reinforcementLearning = new ActorCriticNN(numberOfActions, (int) sceneState.size(), &stateAnalyzer);

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
		int changeController = 3;
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
		}
		if(analyzeResult.additionalInfo == ScenarioAdditionalInfo::notFound)
					throw std::string("Could not initialize, check player visibility");

		sceneState = stateAnalyzer.createSceneState(analyzeResult.processedImage,
													analyzeResult.processedImagePast,
													analyzeResult.processedImagePast2,
													controllerInput,
													analyzeResult.playerCoords,
													analyzeResult.playerVelocity);
		action = reinforcementLearning->chooseAction(sceneState);
		controllerInput = determineControllerInput(action);
		MemoryAnalyzer::getPtr()->setController(determineControllerInputInt(action));

		while(1)
		{
			//Persist info
#ifdef PRINT_PROCESSING_TIME
			int64 timeBefore = cv::getTickCount();
#endif
			cv::waitKey(80);
			State oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = stateAnalyzer.analyze();
			if(analyzeResult.processedImage.cols == 0) continue;
			sceneState = stateAnalyzer.createSceneState(analyzeResult.processedImage,
														analyzeResult.processedImagePast,
														analyzeResult.processedImagePast2,
														controllerInput,
														analyzeResult.playerCoords,
														analyzeResult.playerVelocity);
			if(analyzeResult.reward >= StateAnalyzer::LITTLE_ADVANCE_REWARD ) score++ ;

//			DataDrawer::drawReducedState(sceneState, &stateAnalyzer);
//			DataDrawer::drawAdditionalInfo(analyzeResult.reward, TIME_LIMIT, time, controllerInput, sceneState[sceneState.size()-1]);

			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));

			//Determine new controller input
			changeController--;
			if(changeController < 1)
			{
				changeController = 1;
				action = reinforcementLearning->chooseAction(sceneState);
				controllerInput = determineControllerInput(action);
				MemoryAnalyzer::getPtr()->setController(determineControllerInputInt(action));
			}

			//Draw info
//			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extraxtedSceneData = extractSceneState(sceneState);
//			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> redExtraxtedSceneData = extractSceneState(reduceSceneState(sceneState,0),8,14);
//			DataDrawer::drawAnalyzedData(extraxtedSceneData.first,extraxtedSceneData.second,
//					analyzeResult.reward,0);
//			DataDrawer::drawAnalyzedData(redExtraxtedSceneData.first,redExtraxtedSceneData.second,
//					analyzeResult.reward,0);
//			LogFileHandler::printState(sceneState);

#ifdef PRINT_PROCESSING_TIME
			int64 afterBefore = cv::getTickCount();
			std::cout << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
#endif
			//Timer
			if(analyzeResult.reward < StateAnalyzer::LITTLE_ADVANCE_REWARD) time--;
			else if(analyzeResult.reward > StateAnalyzer::LITTLE_ADVANCE_REWARD) time+=4;
			if(time > TIME_LIMIT) time = TIME_LIMIT;

			//End?
			if(analyzeResult.endScenario || time<0)
			{
				if(time<0) historyScenario.begin()->reward = 0.5; // No kill penalty for timeout
				scenarioResult = analyzeResult.additionalInfo;
				break;
			}
		}

		std::cout << score << "\n";
		LogFileHandler::logValue("score.log",score);

		//End scenario
		MemoryAnalyzer::getPtr()->setController(0);
		loadParameters();
		stateAnalyzer.correctScenarioHistory(historyScenario, scenarioResult);

		//Learn
		double sumErrHist = reinforcementLearning->learnFromScenario(historyScenario);
		double sumErrMem = reinforcementLearning->learnFromMemory();

		reinforcementLearning->handleParameters();
	}
}

/*
 *
 */
void Bot::loadParameters()
{
	if(ParameterFileHandler::checkParameter("quit.param","Bot::Exit program"))
		throw std::string("Exit program");
}

/*
 *
 */
ControllerInput Bot::determineControllerInput(int t_action)
{
	ControllerInput w;
	for(int i=0; i<numberOfControllerInputs; i++) w.push_back(false);

//	w[ 2+(t_action%4) ] = true;
//	w[0] = t_action>3;
	w[0] = t_action == 1;
	w[3] = true;

	return w;
}

/*
 *
 */
int Bot::determineControllerInputInt(int t_action)
{
//	int direction = (1<<(4+t_action%4));
//	int jump = t_action>3?1:0;
	return (1<<7)+t_action;
}

/*
 *
 */
std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> Bot::extractSceneState(State sceneState, int xScreenSize, int yScreenSize)
{
	std::cout << sceneState.size() << "\n";
	cv::Mat fieldAndEnemiesLayout = cv::Mat(yScreenSize, xScreenSize, CV_8UC3);

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
	result.first.processedImage = fieldAndEnemiesLayout;

//	//AdditionalInfo
	result.first.playerCoords.x = sceneState[sceneState.size()-4];
	result.first.playerCoords.y = sceneState[sceneState.size()-3];
	result.first.playerVelocity.x = sceneState[sceneState.size()-2];
	result.first.playerVelocity.y = sceneState[sceneState.size()-1];

	//Controller
//	for(int i=0;i<6; i++) result.second.push_back(sceneState[sceneState.size()-1+i]);

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
