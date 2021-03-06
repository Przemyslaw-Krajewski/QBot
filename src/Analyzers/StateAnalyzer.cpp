/*
 * StateAnalyzer.cpp
 *
 *  Created on: 7 lut 2019
 *      Author: mistrz
 */

#include "StateAnalyzer.h"

StateAnalyzer::StateAnalyzer()
{
	game = Game::SuperMarioBros;
	imageAnalyzer = new RawImageAnalyzer(game);
}

StateAnalyzer::~StateAnalyzer()
{
	delete imageAnalyzer;
}

StateAnalyzer::AnalyzeResult StateAnalyzer::analyze()
{
	if(game==Game::SuperMarioBros) return analyzeSMB();
	else if(game==Game::BattleToads) return analyzeBT();
	else throw std::string("StateAnalyzer::No such game");
}

State StateAnalyzer::createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
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
	if(!imageAnalyzeResult.playerFound)			     										 	 {reward = NOTHING_REWARD;std::cout << "Not found\n";}
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy)				 {reward = DIE_REWARD; endScenario = true;}
	else if(imageAnalyzeResult.playerWon) 			 									 		 {reward = WIN_REWARD; endScenario = true;std::cout << "Win\n";}
	else if(memoryAnalyzeResult.playerPositionY > 180) 											 {reward = DIE_REWARD; endScenario = true;} //pitfall
	else if(memoryAnalyzeResult.playerPositionX < 20) 											 {reward = DIE_REWARD; endScenario = true;} // left border
	else if(memoryAnalyzeResult.playerPositionX > 96 && memoryAnalyzeResult.playerVelocityX > 16){reward = ADVANCE_REWARD;}
	else if(memoryAnalyzeResult.playerVelocityX > 8) 										 	 {reward = LITTLE_ADVANCE_REWARD;}

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.processedImage = imageAnalyzeResult.processedImage;
	analyzeResult.processedImagePast = imageAnalyzeResult.processedImagePast;
	analyzeResult.additionalInfo = additionalInfo;
	analyzeResult.playerCoords = Point(memoryAnalyzeResult.playerPositionX,memoryAnalyzeResult.playerPositionY);
	analyzeResult.playerVelocity = Point(memoryAnalyzeResult.playerVelocityX,memoryAnalyzeResult.playerVelocityY);
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
