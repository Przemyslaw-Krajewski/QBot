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
	deathReason = DeathReason::noInfo;
	time = 0;

	/*  Initialize */
	cv::waitKey(3000);
	//Load game in order to avoid not finding player during initializing
	DesktopHandler::getPtr()->releaseControllerButton();
	DesktopHandler::loadGame();

	//push 6 controller inputs
	for(int i=0; i<6; i++) controllerInput.push_back(false);

	//Initialize scene data
	StateAnalyzer::AnalyzeResult analyzeResult;
	for(int i=1; i<11; i++)
	{
		analyzeResult = analyzer.analyze();
		if(analyzeResult.additionalInfo != StateAnalyzer::AnalyzeResult::notFound) break;
		cv::waitKey(1000);
		std::cout << "Could not find player, atteption: " << i << "\n";
	}
	if(analyzeResult.additionalInfo == StateAnalyzer::AnalyzeResult::notFound)
				throw std::string("Could not initialize, check player visibility");

	sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
				 	 	 	 	  controllerInput,
								  analyzeResult.playerCoords,
								  analyzeResult.playerVelocity);
	StateAnalyzer::printAnalyzeData(analyzeResult);
	cv::waitKey(1000);

	//Initialize qLearning
	qLearning = new QLearning(5, std::vector<int>(sceneState.size(),1), hashmap);
}

/*
 *
 */
Bot::~Bot()
{
	delete qLearning;
}

/*
 *
 */
std::vector<bool> Bot::determineControllerInput(int t_action)
{
	std::vector<bool> w;
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);

	switch(t_action)
	{
	case 0: //Right
		w[4] = true;
		break;
	case 1: //Right jump
		w[0] = true;
		w[4] = true;
		break;
	case 2: //Left
		w[2] = true;
		break;
	case 3: //Jump
		w[0] = true;
		break;
	case 4: //Left Jump
		w[0] = true;
		w[2] = true;
		break;
	default:
		std::cout << t_action << "\n";
		assert("No such action!");
	}

	return w;
}

/*
 *
 */
std::vector<int> Bot::createSceneState(cv::Mat& fieldAndEnemiesLayout, std::vector<bool>& controllerInput, StateAnalyzer::Point& position, StateAnalyzer::Point& velocity)
{
	std::vector<int> sceneState;

	//Terrain
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(ptr[0]==100 && ptr[1]==100 && ptr[2]==100) sceneState.push_back(MAX_INPUT_VALUE);
			else sceneState.push_back(MIN_INPUT_VALUE);
		}
	}
	//Enemies
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(ptr[0]==0 && ptr[1]==0 && ptr[2]==220) sceneState.push_back(MAX_INPUT_VALUE);
			else sceneState.push_back(MIN_INPUT_VALUE);
		}
	}
	//Controller
	for(bool ci : controllerInput)
	{
		if(ci==true) sceneState.push_back(MAX_INPUT_VALUE);
		else sceneState.push_back(MIN_INPUT_VALUE);
	}
	//AdditionalInfo
	sceneState.push_back(position.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(position.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);

	return sceneState;
}

/*
 *
 */
void Bot::run()
{
	while(1)
	{
		prepareGameBeforeRun();
		deathReason = DeathReason::noInfo;
		time = TIME_LIMIT;
		action = 0;

		//Get first state
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
		sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
						controllerInput,
						analyzeResult.playerCoords,
						analyzeResult.playerVelocity);
		while(1)
		{
			std::vector<int> oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
			sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
						 	 	 	 	  controllerInput,
										  analyzeResult.playerCoords,
										  analyzeResult.playerVelocity);

			//add learning info to history
			historyScenario.push_front(HistoryEntry(oldSceneState, sceneState, oldAction, analyzeResult.reward));
			if(sceneState.size() == 906) qLearning->addDiscoveredState(sceneState);
//			else assert("Bad size" && false);

			//Determine new controller input
			action = qLearning->chooseAction(sceneState).second;
			controllerInput = determineControllerInput(action);
			DesktopHandler::getPtr()->pressControllerButton(controllerInput);

			//Print info
			StateAnalyzer::printAnalyzeData(analyzeResult);
			for(int i=0; i<5; i++) std::cout << (int) qLearning->getQValue(oldSceneState,i) << " ";
			std::cout << ": " << analyzeResult.reward << "   " << time << "\n";
			std::cout << analyzeResult.playerVelocity.x << "  " << analyzeResult.playerVelocity.y << "\n";

			//Stop game?
			if(manageScenarioTime(analyzeResult.reward==50)) break;
			if(analyzeResult.endScenario)
			{
				historyScenario.pop_front();
				historyScenario.front().reward=-1000;
				deathReason = analyzeResult.additionalInfo;
				break;
			}
		}
		printDeathReason();
		learnQLearningScenario();
		DesktopHandler::getPtr()->releaseControllerButton();
	}
}

/*
 *
 */
void Bot::prepareGameBeforeRun()
{
	DesktopHandler::getPtr()->releaseControllerButton();
	DesktopHandler::loadGame();
	cv::waitKey(3000);
	historyScenario.clear();
}

/*
 * @return true if is time out
 */
bool Bot::manageScenarioTime(bool resetTimer)
{
	if(!resetTimer)
	{
		time--;
		if(time<0)
		{
			deathReason = DeathReason::timeOut;
			return true;
		}
		return false;
	}
	else
	{
		time = TIME_LIMIT;
		return false;
	}
}

/*
 *
 */
void Bot::learnQLearningScenario()
{
	double change = 0;
	long numberOfProbes = 0;
	for(std::list<HistoryEntry>::iterator historyIterator = historyScenario.begin(); historyIterator!=historyScenario.end(); historyIterator++)
	{
		change += fabs(qLearning->learn(historyIterator->oldState, historyIterator->state, historyIterator->action, historyIterator->reward));
		numberOfProbes++;
	}
	std::cout << "Change:" << change/numberOfProbes << "\n";
}

/*
 *
 */
void Bot::printDeathReason()
{
	//Print death reason
	switch(deathReason)
	{
		case DeathReason::timeOut:
			std::cout << "TimeOut" << "\n"; break;
		case DeathReason::killedByEnemy:
			std::cout << "Enemy" << "\n"; break;
		case DeathReason::fallenInPitfall:
			std::cout << "Pitfall " << "\n"; break;
		default:
			std::cout << "????" << "\n"; break;
	}
}

/*
 *
 */
void Bot::testStateAnalyzer()
{
	while(1)
	{
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
		//Print info
		StateAnalyzer::printAnalyzeData(analyzeResult);
		std::cout << ": " << analyzeResult.reward << "   " << time << "\n";
		std::cout << analyzeResult.playerVelocity.x << "  " << analyzeResult.playerVelocity.y << "\n";
	}
}
