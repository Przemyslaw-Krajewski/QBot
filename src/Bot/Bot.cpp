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
	/*  Initialize */
	cv::waitKey(1000);
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

	Controller controller = Controller(0);
	State sceneState = stateAnalyzer.createSceneState(analyzeResult.processedImage,
													  analyzeResult.processedImagePast,
													  analyzeResult.processedImagePast2,
													  controller.getInput(),
													  analyzeResult.playerCoords,
													  analyzeResult.playerVelocity);
	//Initialize acLearning
	reinforcementLearning = new ActorCriticNN(numberOfActions, (int) sceneState.size(), &stateAnalyzer);
	reinforcementLearning->handleParameters();
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
		//Reset variables
		historyScenario.clear();
		double score = 0;
		int time = TIME_LIMIT;
		ScenarioAdditionalInfo scenarioResult = ScenarioAdditionalInfo::noInfo;

		//Reload game
		MemoryAnalyzer::getPtr()->setController(0);
		MemoryAnalyzer::getPtr()->loadState();
		cv::waitKey(100);

		//Get first state
		StateAnalyzer::AnalyzeResult analyzeResult = stateAnalyzer.analyze();
		state = stateAnalyzer.createSceneState(analyzeResult.processedImage,
													analyzeResult.processedImagePast,
													analyzeResult.processedImagePast2,
													controller.getInput(),
													analyzeResult.playerCoords,
													analyzeResult.playerVelocity);

		controller = Controller(reinforcementLearning->chooseAction(state));
		MemoryAnalyzer::getPtr()->setController(controller.getCode());

		while(1)
		{
#ifdef PRINT_PROCESSING_TIME
			int64 timeBefore = cv::getTickCount();
#endif
			//Persist info
			prevState = state;
			prevController=controller;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = stateAnalyzer.analyze();
			state = stateAnalyzer.createSceneState(analyzeResult.processedImage,
														analyzeResult.processedImagePast,
														analyzeResult.processedImagePast2,
														prevController.getInput(),
														analyzeResult.playerCoords,
														analyzeResult.playerVelocity);
			if(analyzeResult.reward >= StateAnalyzer::LITTLE_ADVANCE_REWARD ) score++ ;

//			DataDrawer::drawReducedState(sceneState, &stateAnalyzer);
			DataDrawer::drawAdditionalInfo(analyzeResult.reward, TIME_LIMIT, time, controller.getInput(), state[state.size()-1]);

			//add learning info to history
			historyScenario.push_front(SARS(prevState, state, prevController.getAction(), analyzeResult.reward));

			//Determine new controller input
			controller = Controller(reinforcementLearning->chooseAction(state));
			MemoryAnalyzer::getPtr()->setController(controller.getCode());

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

			cv::waitKey(60);
		}

		std::cout << score << "\n";
		LogFileHandler::logValue("score.log",score);

		//End scenario
		MemoryAnalyzer::getPtr()->setController(0);
		handleParameters();

		//Learn
		stateAnalyzer.correctScenarioHistory(historyScenario, scenarioResult);
		double sumErrHist = reinforcementLearning->learnFromScenario(historyScenario);
		double sumErrMem = reinforcementLearning->learnFromMemory();

		reinforcementLearning->handleParameters();
	}
}

/*
 *
 */
void Bot::handleParameters()
{
	if(ParameterFileHandler::checkParameter("quit.param","Bot::Exit program"))
		throw std::string("Exit program");
}
