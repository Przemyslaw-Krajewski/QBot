/*
 * StateAnalyzer.cpp
 *
 *  Created on: 7 lut 2019
 *      Author: mistrz
 */

#include "StateAnalyzer.h"

StateAnalyzer::StateAnalyzer()
{
	game = Game::BattleToads;
	imageAnalyzer = new RawImageAnalyzer(game);
}

StateAnalyzer::~StateAnalyzer()
{

}

StateAnalyzer::AnalyzeResult StateAnalyzer::analyze()
{
	if(game==Game::SuperMarioBros) return analyzeSMB();
	else if(game==Game::BattleToads) return analyzeBT();
	else throw std::string("StateAnalyzer::No such game");
}

std::vector<int> StateAnalyzer::createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
											  ControllerInput& controllerInput, Point& position, Point& velocity)
{
	return imageAnalyzer->createSceneState(image, imagePast, imagePast2, controllerInput, position, velocity);
}

StateAnalyzer::AnalyzeResult StateAnalyzer::analyzeSMB()
{
//	int64 timeBefore = cv::getTickCount();
	//Get Desktop Screen
	cv::Mat colorImage = MemoryAnalyzer::getPtr()->fetchScreenData();

	ImageAnalyzer::AnalyzeResult imageAnalyzeResult;
	imageAnalyzer->processImage(&colorImage, &imageAnalyzeResult);
	MemoryAnalyzer::AnalyzeResult memoryAnalyzeResult = MemoryAnalyzer::getPtr()->fetchData();

//	int64 timeAfter = cv::getTickCount();
//	std::cout << (timeAfter - timeBefore)/ cv::getTickFrequency() << "\n";

	//Additional info
	ScenarioAdditionalInfo additionalInfo = ScenarioAdditionalInfo::noInfo;
	if(!imageAnalyzeResult.playerFound) additionalInfo = ScenarioAdditionalInfo::notFound;
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy) additionalInfo = ScenarioAdditionalInfo::killedByEnemy;
	else if(imageAnalyzeResult.playerIsDead) additionalInfo = ScenarioAdditionalInfo::fallenInPitfall;
	else if(imageAnalyzeResult.playerWon) additionalInfo = ScenarioAdditionalInfo::won;

	//reward
	double reward = NOTHING_REWARD;
	bool endScenario = false;
	if(!imageAnalyzeResult.playerFound)			     										 	 {reward = 0.0001;  endScenario = true;}
	else if(imageAnalyzeResult.playerIsDead) 			 									 	 {reward = DIE_REWARD; endScenario = true;}
	else if(imageAnalyzeResult.playerWon) 			 									 		 {reward = WIN_REWARD; endScenario = true;}
	else if(memoryAnalyzeResult.playerPositionX > 60 && memoryAnalyzeResult.playerVelocityX > 16){reward = ADVANCE_REWARD;}
	else if(memoryAnalyzeResult.playerVelocityX > 16) 										 	 {reward = LITTLE_ADVANCE_REWARD;}

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.processedImage = imageAnalyzeResult.processedImage;
	analyzeResult.processedImagePast = imageAnalyzeResult.processedImagePast;
	analyzeResult.additionalInfo = additionalInfo;
	analyzeResult.playerCoords = Point(memoryAnalyzeResult.playerPositionX,memoryAnalyzeResult.playerPositionY);
	analyzeResult.playerVelocity = Point(memoryAnalyzeResult.playerVelocityX/4,memoryAnalyzeResult.playerVelocityY/2);
	analyzeResult.reward = reward;
	analyzeResult.endScenario = endScenario;

	return analyzeResult;
}

StateAnalyzer::AnalyzeResult StateAnalyzer::analyzeBT()
{

//	int64 timeBefore = cv::getTickCount();
	//Get Desktop Screen
	cv::Mat colorImage = MemoryAnalyzer::getPtr()->fetchScreenData();

	ImageAnalyzer::AnalyzeResult imageAnalyzeResult;
	imageAnalyzer->processImage(&colorImage, &imageAnalyzeResult);

//	int64 timeAfter = cv::getTickCount();
//	std::cout << (timeAfter - timeBefore)/ cv::getTickFrequency() << "\n";

//	std::cout << (int) MemoryAnalyzer::getPtr()->getMemValue(0x555555ac1408) << "\n";
//	MemoryAnalyzer::getPtr()->setMemValue(0x555555ac1408,32);

	//Additional info
	ScenarioAdditionalInfo additionalInfo = ScenarioAdditionalInfo::noInfo;
	if(!imageAnalyzeResult.playerFound) additionalInfo = ScenarioAdditionalInfo::notFound;
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy) additionalInfo = ScenarioAdditionalInfo::killedByEnemy;
	else if(imageAnalyzeResult.playerIsDead) additionalInfo = ScenarioAdditionalInfo::fallenInPitfall;
	else if(imageAnalyzeResult.playerWon) additionalInfo = ScenarioAdditionalInfo::won;

	//reward
	double reward = ADVANCE_REWARD;
	bool endScenario = false;
	if(imageAnalyzeResult.playerWon)		 {reward = WIN_REWARD; endScenario = true;}
	else if(imageAnalyzeResult.playerIsDead) {reward = DIE_REWARD; endScenario = true;}

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.processedImage = imageAnalyzeResult.processedImage;
	analyzeResult.processedImagePast = imageAnalyzeResult.processedImagePast;
	analyzeResult.processedImagePast2 = imageAnalyzeResult.processedImagePast2;
	analyzeResult.additionalInfo = additionalInfo;
	analyzeResult.reward = reward;
	analyzeResult.endScenario = endScenario;

	return analyzeResult;
}
