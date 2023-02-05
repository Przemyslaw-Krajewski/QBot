/*
 * StateAnalyzer.cpp
 *
 *  Created on: 7 lut 2019
 *      Author: mistrz
 */

#include "StateAnalyzer.h"

/*
 *
 */
StateAnalyzer::StateAnalyzer()
{
	timeLimit = TIME_LIMIT;
	game = Game::SuperMarioBros;
	imageAnalyzer = new RawImageAnalyzer(game);
}

/*
 *
 */
StateAnalyzer::~StateAnalyzer()
{
	delete imageAnalyzer;
}

/*
 *
 */
StateAnalyzer::AnalyzeResult StateAnalyzer::analyze(ControllerInput &t_input)
{
	//Analyze
	StateAnalyzer::AnalyzeResult result;
	if(game==Game::SuperMarioBros) result=analyzeSMB(t_input);
	else if(game==Game::BattleToads) result=analyzeBT(t_input);
	else throw std::string("StateAnalyzer::No such game");

	//Timer
	if(handleTimeLimit(result.reward))
	{
		result.reward = TIMEOUT_REWARD;
		result.scenarioStatus=ScenarioAdditionalInfo::timeOut;
	}

#ifdef PRINT_REDUCED_IMAGE
	imageAnalyzer->getReduceStateMethod()(result.processedState);
#endif

	//Draw
	DataDrawer::drawAdditionalInfo(result.reward,
								   StateAnalyzer::TIME_LIMIT,
								   timeLimit,
								   t_input,
								   result.processedState[result.processedState.size()-1]>0);

	return result;
}

/*
 *
 */
StateAnalyzer::AnalyzeResult StateAnalyzer::analyzeSMB(ControllerInput &t_input)
{
	//Get Desktop Screen
	cv::Mat colorImage = MemoryAnalyzer::getPtr()->fetchScreenData();

	ImageAnalyzer::AnalyzeResult imageAnalyzeResult;
	imageAnalyzer->processImage(&colorImage, &imageAnalyzeResult);
	MemoryAnalyzer::AnalyzeResult memoryAnalyzeResult = MemoryAnalyzer::getPtr()->fetchData();

	//Additional info and reward
	ScenarioAdditionalInfo additionalInfo = ScenarioAdditionalInfo::ok;
	double reward = NOTHING_REWARD;

	if(!imageAnalyzeResult.playerFound)			     										 	 {reward = NOTHING_REWARD;		  additionalInfo = ScenarioAdditionalInfo::ok;}
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy)				 {reward = DIE_REWARD; 	  		  additionalInfo = ScenarioAdditionalInfo::killedByEnemy;}
	else if(imageAnalyzeResult.playerWon) 			 									 		 {reward = WIN_REWARD;    		  additionalInfo = ScenarioAdditionalInfo::won;}
	else if(score != memoryAnalyzeResult.score)													 {reward = GREAT_ADVANCE_REWARD;  additionalInfo = ScenarioAdditionalInfo::ok;}
	else if(memoryAnalyzeResult.playerPositionX == 0 && memoryAnalyzeResult.playerVelocityY <= 0){reward = ADVANCE_REWARD;    	  additionalInfo = ScenarioAdditionalInfo::ok;}  //pipe
	else if(memoryAnalyzeResult.playerPositionY > 180)											 {reward = DIE_REWARD;    		  additionalInfo = ScenarioAdditionalInfo::killedByEnvironment;}  //pitfall
	else if(memoryAnalyzeResult.playerPositionX < 20)											 {reward = DIE_REWARD;    		  additionalInfo = ScenarioAdditionalInfo::killedByEnvironment;}  // left border
	else if(memoryAnalyzeResult.screenVelocity == 2)											 {reward = ADVANCE_REWARD;		  additionalInfo = ScenarioAdditionalInfo::ok;}
//	else if(screenPosition > (memoryAnalyzeResult.screenPosition & 127))						 {reward = CHECKPOINT_REWARD;	  additionalInfo = ScenarioAdditionalInfo::ok;}
	else if(memoryAnalyzeResult.screenVelocity > 0) 										 	 {reward = ADVANCE_REWARD; 		  additionalInfo = ScenarioAdditionalInfo::ok;}

	screenPosition = (memoryAnalyzeResult.screenPosition & 127);
	score = memoryAnalyzeResult.score;

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.scenarioStatus = additionalInfo;
	analyzeResult.playerCoords = Point(memoryAnalyzeResult.playerPositionX,memoryAnalyzeResult.playerPositionY);
	analyzeResult.playerVelocity = Point(memoryAnalyzeResult.playerVelocityX,memoryAnalyzeResult.playerVelocityY);
	analyzeResult.reward = reward;
	analyzeResult.processedState = imageAnalyzer->createSceneState(imageAnalyzeResult.processedImages,
																   t_input,
																   analyzeResult.playerCoords,
																   analyzeResult.playerVelocity);

	return analyzeResult;
}

/*
 *
 */
StateAnalyzer::AnalyzeResult StateAnalyzer::analyzeBT(ControllerInput &t_input)
{
	//Get Desktop Screen
	cv::Mat colorImage = MemoryAnalyzer::getPtr()->fetchScreenData();

	ImageAnalyzer::AnalyzeResult imageAnalyzeResult;
	imageAnalyzer->processImage(&colorImage, &imageAnalyzeResult);
	MemoryAnalyzer::AnalyzeResult memoryAnalyzeResult = MemoryAnalyzer::getPtr()->fetchDataBT();

	//Additional info
	ScenarioAdditionalInfo additionalInfo = ScenarioAdditionalInfo::ok;
	double reward = ADVANCE_REWARD;
	if(!imageAnalyzeResult.playerFound) additionalInfo = ScenarioAdditionalInfo::playerNotFound;
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy){reward = DIE_REWARD; additionalInfo = ScenarioAdditionalInfo::killedByEnemy;}
	else if(imageAnalyzeResult.playerIsDead) 									{reward = DIE_REWARD; additionalInfo = ScenarioAdditionalInfo::killedByEnvironment;}
	else if(memoryAnalyzeResult.playerPositionY < 0x40)							{reward = DIE_REWARD; additionalInfo = ScenarioAdditionalInfo::killedByEnvironment;}
	else if(imageAnalyzeResult.playerWon) 										{reward = WIN_REWARD; additionalInfo = ScenarioAdditionalInfo::won;}
	else if((memoryAnalyzeResult.playerPositionX < 0xD5 && memoryAnalyzeResult.obstaclePositionX == 0xC8 ||
			memoryAnalyzeResult.playerPositionX > 0xD5 && memoryAnalyzeResult.obstaclePositionX == 0xE0) &&
			memoryAnalyzeResult.obstaclePositionZ < 0x30 && memoryAnalyzeResult.obstaclePositionZ != 0x0 )
																				{reward = DIE_REWARD; additionalInfo = ScenarioAdditionalInfo::killedByEnvironment;}

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.scenarioStatus = additionalInfo;
	analyzeResult.playerCoords = Point(memoryAnalyzeResult.playerPositionX,memoryAnalyzeResult.playerPositionY);
	analyzeResult.reward = reward;

	Point p = Point();
	analyzeResult.processedState = imageAnalyzer->createSceneState(imageAnalyzeResult.processedImages, t_input, analyzeResult.playerCoords ,p);

	return analyzeResult;
}

/*
 *
 */
bool StateAnalyzer::handleTimeLimit(double t_gainedReward)
{
	if(t_gainedReward < LITTLE_ADVANCE_REWARD) timeLimit--;
	else if(t_gainedReward > ADVANCE_REWARD) timeLimit = TIME_LIMIT;
	else if(t_gainedReward >= LITTLE_ADVANCE_REWARD) timeLimit+=4;
	if(timeLimit > TIME_LIMIT) timeLimit = TIME_LIMIT;

	return timeLimit<0;
}

/*
 *
 */
void StateAnalyzer::correctScenarioHistory(std::list<SARS> &t_history, ScenarioAdditionalInfo t_additionalInfo)
{
	if(t_additionalInfo==ScenarioAdditionalInfo::timeOut && t_history.size()>5)
	{
		int erased = 0;
		bool isBlocked = false;
		std::list<SARS>::iterator toErase = t_history.begin();
		std::list<SARS>::iterator it = t_history.begin();it++;it++;it++;it++;
		for( ; it != t_history.end();it++)
		{
			if((it->reward == NOTHING_REWARD || it->reward == TIMEOUT_REWARD) && (it->action != 2 || isBlocked))
			{
				if(it->action != 2) isBlocked = true;
				toErase = t_history.erase(toErase);
				erased++;
			}
			else break;
		}
		t_history.begin()->reward = TIMEOUT_REWARD;
		std::cout << "Timeout Erased:" << erased << "\n";
	}
	imageAnalyzer->correctScenarioHistory(t_history, t_additionalInfo==ScenarioAdditionalInfo::killedByEnemy);
}
