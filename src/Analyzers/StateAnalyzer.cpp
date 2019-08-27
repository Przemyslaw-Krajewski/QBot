/*
 * StateAnalyzer.cpp
 *
 *  Created on: 7 lut 2019
 *      Author: mistrz
 */

#include "StateAnalyzer.h"

StateAnalyzer::StateAnalyzer()
{

}

StateAnalyzer::~StateAnalyzer()
{

}

StateAnalyzer::AnalyzeResult StateAnalyzer::analyze()
{
	//Get Desktop Screen
	cv::Mat image = MemoryAnalyzer::getPtr()->fetchScreenData();

	ImageAnalyzer::AnalyzeResult imageAnalyzeResult = imageAnalyzer.processImage(&image);
	MemoryAnalyzer::AnalyzeResult memoryAnalyzeResult = MemoryAnalyzer::getPtr()->fetchData();

	//Additional info
	AnalyzeResult::AdditionalInfo additionalInfo;
	if(!imageAnalyzeResult.playerFound) additionalInfo = AnalyzeResult::AdditionalInfo::notFound;
	else if(imageAnalyzeResult.playerIsDead && imageAnalyzeResult.killedByEnemy) additionalInfo = AnalyzeResult::AdditionalInfo::killedByEnemy;
	else if(imageAnalyzeResult.playerIsDead) additionalInfo = AnalyzeResult::AdditionalInfo::fallenInPitfall;
	else if(imageAnalyzeResult.playerWon) additionalInfo = AnalyzeResult::AdditionalInfo::won;

	//reward
	double reward = 0.1;
	bool endScenario = false;
	if(!imageAnalyzeResult.playerFound)			     										 	 {reward = 0.0001;  endScenario = true;}
	else if(imageAnalyzeResult.playerIsDead) 			 									 	 {reward = DIE_REWARD; endScenario = true;}
	else if(imageAnalyzeResult.playerWon) 			 									 		 {reward = WIN_REWARD; endScenario = true;}
	else if(memoryAnalyzeResult.playerPositionX > 60 && memoryAnalyzeResult.playerVelocityX > 16){reward = ADVANCE_REWARD;}
	else if(memoryAnalyzeResult.playerVelocityX > 16) 										 	 {reward = LITTLE_ADVANCE_REWARD;}

	//Preparing output
	AnalyzeResult analyzeResult;
	analyzeResult.fieldAndEnemiesLayout = imageAnalyzeResult.fieldAndEnemiesLayout;
	analyzeResult.additionalInfo = additionalInfo;
	analyzeResult.playerCoords = Point(0,0);
	analyzeResult.playerVelocity = Point(memoryAnalyzeResult.playerVelocityX/4,memoryAnalyzeResult.playerVelocityY/2);
	analyzeResult.reward = reward;
	analyzeResult.endScenario = endScenario;

	return analyzeResult;
}
