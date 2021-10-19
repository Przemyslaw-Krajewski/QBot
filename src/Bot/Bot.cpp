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
		analyzeResult = stateAnalyzer.analyze(controller.getInput());
		if(analyzeResult.isPlayerFound()) break;
		cv::waitKey(1000);
		std::cout << "Could not find player, atteption: " << i << "\n";
	}
	if(analyzeResult.scenarioStatus == ScenarioAdditionalInfo::playerNotFound)
				throw std::string("Could not initialize, check player visibility");

	Controller controller = Controller(0);
	State sceneState = analyzeResult.processedState;
	//Initialize acLearning
	reinforcementLearning = new ActorCriticNN(numberOfActions, sceneState.size(), stateAnalyzer.getReduceStateMethod());
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
		stateAnalyzer.resetTimeLimit();
		ScenarioAdditionalInfo scenarioResult = ScenarioAdditionalInfo::ok;

		//Reload game
		MemoryAnalyzer::getPtr()->setController(0);
		MemoryAnalyzer::getPtr()->loadState();
		cv::waitKey(1000);

		//Get first state
		analyzeResult = stateAnalyzer.analyze(controller.getInput());

		controller = Controller(reinforcementLearning->chooseAction(analyzeResult.processedState));
		MemoryAnalyzer::getPtr()->setController(controller.getCode());

		while(1)
		{
			//Persist prev data
			prevAnalyzeResult = analyzeResult;
			prevController=controller;

			//Analyze situation
			analyzeResult = stateAnalyzer.analyze(controller.getInput());
			if(analyzeResult.reward >= StateAnalyzer::LITTLE_ADVANCE_REWARD ) score++ ;

			//add learning info to history
			historyScenario.push_front(SARS(prevAnalyzeResult.processedState,
											analyzeResult.processedState,
											prevController.getAction(),
											analyzeResult.reward));

			//Determine new controller input
			controller = Controller(reinforcementLearning->chooseAction(analyzeResult.processedState));
			MemoryAnalyzer::getPtr()->setController(controller.getCode());

			//End?
			if(analyzeResult.endScenario()) break;
			cv::waitKey(80);
		}

		std::cout << "Achieved score: "<< score << "\n";
		LogFileHandler::logValue("score.log",score);

		//End scenario
		MemoryAnalyzer::getPtr()->setController(0);
		handleParameters();

		//Learn
		stateAnalyzer.correctScenarioHistory(historyScenario, analyzeResult.scenarioStatus);
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
