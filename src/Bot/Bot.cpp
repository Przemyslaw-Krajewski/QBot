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
//	DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
	cv::waitKey(1000);

	//Initialize acLearning
	reinforcementLearning = new ActorCriticNN(numberOfActions, (int) sceneState.size());

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
			if(analyzeResult.processedImage.cols == 0) continue;
			if(analyzeResult.reward >= StateAnalyzer::LITTLE_ADVANCE_REWARD ) score++ ;
			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));

			//Determine new controller input
			action = reinforcementLearning->chooseAction(sceneState);
			controllerInput = determineControllerInput(action);
			MemoryAnalyzer::getPtr()->setController(determineControllerInputInt(action));

			//Draw info
			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extraxtedSceneData = extractSceneState(sceneState);
			DataDrawer::drawAnalyzedData(extraxtedSceneData.first,extraxtedSceneData.second,
					analyzeResult.reward,0);

#ifdef PRINT_PROCESSING_TIME
			int64 afterBefore = cv::getTickCount();
			std::cout << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
#endif
			//Timer
			if(analyzeResult.reward < StateAnalyzer::LITTLE_ADVANCE_REWARD)
				time--;
			else if(analyzeResult.reward > StateAnalyzer::LITTLE_ADVANCE_REWARD && time < TIME_LIMIT) time++;

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

		double sumErrHist = reinforcementLearning->learnFromScenario(historyScenario);
		double sumErrMem = reinforcementLearning->learnFromMemory();

		std::cout << "Learn result: " << sumErrHist << "  " << sumErrMem << "\n";
	}
}

/*
 *
 */
void Bot::loadParameters()
{
	if(ParameterFileHandler::checkParameter("ql.param","QL mode activated"))
		controlMode = ControlMode::QL;

	if(ParameterFileHandler::checkParameter("nn.param","NN mode activated"))
		controlMode = ControlMode::NN;

	if(ParameterFileHandler::checkParameter("nnnolearn.param","NN no learn mode activated"))
		controlMode = ControlMode::NNNoLearn;

	if(ParameterFileHandler::checkParameter("hybrid.param","Hybrid mode activated"))
	{
		controlMode = ControlMode::Hybrid;
		playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
	}

	if(ParameterFileHandler::checkParameter("reset.param","Reset has been ordered"))
		reset = true;

	if(ParameterFileHandler::checkParameter("quit.param","Exit program"))
		throw std::string("Exit program");
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
	result.first.processedImage = fieldAndEnemiesLayout;

//	//AdditionalInfo
	result.first.playerCoords.x = sceneState[sceneState.size()-4];
	result.first.playerCoords.y = sceneState[sceneState.size()-3];
	result.first.playerVelocity.x = sceneState[sceneState.size()-2];
	result.first.playerVelocity.y = sceneState[sceneState.size()-1];

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
