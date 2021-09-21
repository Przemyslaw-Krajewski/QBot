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
		result.reward = 0.5; // TODO hardcoded
		result.scenarioStatus=ScenarioAdditionalInfo::timeOut;
	}

	//Draw
	DataDrawer::drawAdditionalInfo(result.reward,
								   StateAnalyzer::TIME_LIMIT,
								   timeLimit,
								   t_input,
								   result.processedState[result.processedState.size()-1]);

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

	//Additional info
	ScenarioAdditionalInfo additionalInfo = ScenarioAdditionalInfo::ok;
	if(!imageAnalyzeResult.playerFound) additionalInfo = ScenarioAdditionalInfo::playerNotFound;
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy) additionalInfo = ScenarioAdditionalInfo::killedByEnemy;
	else if(imageAnalyzeResult.playerIsDead) additionalInfo = ScenarioAdditionalInfo::killedByEnvironment;
	else if(imageAnalyzeResult.playerWon) additionalInfo = ScenarioAdditionalInfo::won;

	//reward
	double reward = NOTHING_REWARD;
	if(!imageAnalyzeResult.playerFound)			     										 	  reward = NOTHING_REWARD;
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy)				  reward = DIE_REWARD;
	else if(imageAnalyzeResult.playerWon) 			 									 		 {reward = WIN_REWARD;std::cout << "Win\n";}
	else if(memoryAnalyzeResult.playerPositionY > 180) 											  reward = DIE_REWARD;  //pitfall
	else if(memoryAnalyzeResult.playerPositionX < 20) 											  reward = DIE_REWARD;  // left border
	else if(memoryAnalyzeResult.playerPositionX > 96 && memoryAnalyzeResult.playerVelocityX > 16) reward = ADVANCE_REWARD;
	else if(memoryAnalyzeResult.playerVelocityX > 8) 										 	  reward = LITTLE_ADVANCE_REWARD;

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.scenarioStatus = additionalInfo;
	analyzeResult.playerCoords = Point(memoryAnalyzeResult.playerPositionX,memoryAnalyzeResult.playerPositionY);
	analyzeResult.playerVelocity = Point(memoryAnalyzeResult.playerVelocityX,memoryAnalyzeResult.playerVelocityY);
	analyzeResult.reward = reward;
	analyzeResult.processedState = imageAnalyzer->createSceneState(imageAnalyzeResult.processedImages, t_input,analyzeResult.playerCoords ,analyzeResult.playerVelocity);

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

	//Additional info
	ScenarioAdditionalInfo additionalInfo = ScenarioAdditionalInfo::ok;
	if(!imageAnalyzeResult.playerFound) additionalInfo = ScenarioAdditionalInfo::playerNotFound;
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy) additionalInfo = ScenarioAdditionalInfo::killedByEnemy;
	else if(imageAnalyzeResult.playerIsDead) additionalInfo = ScenarioAdditionalInfo::killedByEnvironment;
	else if(imageAnalyzeResult.playerWon) additionalInfo = ScenarioAdditionalInfo::won;

	//reward
	double reward = ADVANCE_REWARD;
	if(imageAnalyzeResult.playerWon)		 reward = WIN_REWARD;
	else if(imageAnalyzeResult.playerIsDead) reward = DIE_REWARD;

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.scenarioStatus = additionalInfo;
	analyzeResult.reward = reward;
	Point p = Point();
	analyzeResult.processedState = imageAnalyzer->createSceneState(imageAnalyzeResult.processedImages, t_input,p ,p);

	return analyzeResult;
}

/*
 *
 */
bool StateAnalyzer::handleTimeLimit(double t_gainedReward)
{
	if(t_gainedReward < LITTLE_ADVANCE_REWARD) timeLimit--;
	else if(t_gainedReward > LITTLE_ADVANCE_REWARD) timeLimit+=4;
	if(timeLimit > TIME_LIMIT) timeLimit = TIME_LIMIT;

	return timeLimit<0;
}