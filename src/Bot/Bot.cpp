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
	for(int i=0; i<50;)
	{
#ifdef ACTORCRITICNN_ONE_ACTION
		i++;
#endif
		//Reset variables
		historyScenario.clear();
		double score = 0;
		stateAnalyzer.resetTimeLimit();
		ScenarioAdditionalInfo scenarioResult = ScenarioAdditionalInfo::ok;
		viewMemory = viewHistory = false;

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
			int64 timeBefore = cv::getTickCount();

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
			historyScenario.begin()->score = score;

			//Determine new controller input
			controller = Controller(reinforcementLearning->chooseAction(analyzeResult.processedState));
			MemoryAnalyzer::getPtr()->setController(controller.getCode());

			int64 timeAfter = cv::getTickCount();
			std::cout << (timeAfter - timeBefore) / cv::getTickFrequency() << "\n";

			//End?
			bool terminate = handleUserInput(cv::waitKey(60));
			if(analyzeResult.endScenario() || terminate) break;
		}

		//End scenario
		std::cout << "Achieved score: "<< score << "\n";
		LogFileHandler::logValue("score.log",score);
		MemoryAnalyzer::getPtr()->setController(0);

		//Prepare history scenario
		stateAnalyzer.correctScenarioHistory(historyScenario, analyzeResult.scenarioStatus);
		reinforcementLearning->prepareScenario(historyScenario, analyzeResult.scenarioStatus==ScenarioAdditionalInfo::timeOut);

		//Learn
		char userInput = -1;
		if(userInput == -1 ) userInput = reinforcementLearning->learnFromScenario(historyScenario).userInput;
		if(userInput == -1 ) userInput = reinforcementLearning->learnFromMemory().userInput;

		reinforcementLearning->handleUserInput(userInput);

		if(viewHistory) StateViewer::viewHistory(&historyScenario);
		if(viewMemory) StateViewer::viewMemory(reinforcementLearning->getMemoryMap());

	}
}

/*
 *
 */
int Bot::handleUserInput(char input)
{
	if(input == 27)
	{
		MemoryAnalyzer::getPtr()->setController(0);
		throw std::string("Exit program");
	}
	else if(input =='t')
	{
		return 1;
	}
	else if(input == 'h')
	{
		std::cout << "History will be viewed\n";
		viewHistory = true;
		return 0;
	}
	else if(input == 'm')
	{
		std::cout << "Memory will be viewed\n";
		viewMemory = true;
		return 0;
	}
	return 0;
}
