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
	cv::Mat image;
	image = cv::Mat(448, 512, CV_8UC3);
	DesktopHandler::getPtr()->getDesktop(&image);

	ImageAnalyzer::AnalyzeResult imageAnalyzeResult = imageAnalyzer.processImage(&image);
	MemoryAnalyzer::AnalyzeResult memoryAnalyzeResult = memoryAnalyzer.fetchData();

	//Additional info
	AnalyzeResult::AdditionalInfo additionalInfo;
	if(!imageAnalyzeResult.playerFound) additionalInfo = AnalyzeResult::AdditionalInfo::notFound;
	else if(imageAnalyzeResult.playerIsDead) additionalInfo = AnalyzeResult::AdditionalInfo::killedByEnemy;

	//reward
	double reward = 0;
	bool endScenario = false;
	if(!imageAnalyzeResult.playerFound)			     										 	 {reward = -100; endScenario = true;}
	else if(imageAnalyzeResult.playerIsDead) 			 									 	 {reward = -1000; endScenario = true;}
	else if(memoryAnalyzeResult.playerPositionX > 99 && memoryAnalyzeResult.playerVelocityX > 6) {reward = 50;}
	else if(memoryAnalyzeResult.playerVelocityX > 5) 										 	 {reward = 10;}

	//Preparing output
	StateAnalyzer::AnalyzeResult analyzeResult;
	analyzeResult.fieldAndEnemiesLayout = imageAnalyzeResult.fieldAndEnemiesLayout;
	analyzeResult.additionalInfo = additionalInfo;
	analyzeResult.playerCoords = Point(0,0);
	analyzeResult.playerVelocity = Point(memoryAnalyzeResult.playerVelocityX/2,memoryAnalyzeResult.playerVelocityY);
	analyzeResult.reward = reward;
	analyzeResult.endScenario = endScenario;

	return analyzeResult;
}

/*
 *
 */
void StateAnalyzer::printAnalyzeData(AnalyzeResult& sceneData)
{
	int blockSize = 15;

	int xScreenSize = sceneData.fieldAndEnemiesLayout.cols;
	int yScreenSize = sceneData.fieldAndEnemiesLayout.rows;

	if(xScreenSize <= 0 && yScreenSize <=0) return;

	cv::Mat map = cv::Mat(blockSize*(yScreenSize+1), blockSize*(xScreenSize), CV_8UC3);

	for(int x=0; x<xScreenSize; x++)
	{
		for(int y=0; y<yScreenSize; y++)
		{
			int fieldValue, enemyValue, playerValue;
			uchar* ptrSrc = sceneData.fieldAndEnemiesLayout.ptr(y)+(x)*3;
			if(ptrSrc[0]==100) {fieldValue=1;enemyValue=0;playerValue=0;}
			else if(ptrSrc[2]==220) {fieldValue=0;enemyValue=1;playerValue=0;}
			else if(ptrSrc[2]==255) {fieldValue=0;enemyValue=0;playerValue=1;}
			else {fieldValue=0;enemyValue=0;playerValue=0;}

			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptr = map.ptr(y*blockSize+yy)+(x*blockSize+xx)*3;
					if(fieldValue == 1) {ptr[0] = 100;ptr[1] = 100;ptr[2] = 100;}
					else if(enemyValue == 1) {ptr[0] = 0;ptr[1] = 0;ptr[2] = 220;}
					else if(playerValue == 1) {ptr[0] = 255;ptr[1] = 255;ptr[2] = 255;}
					else {ptr[0] = 0;ptr[1] = 0;ptr[2] = 0;}
				}
			}
		}
	}
//	if(containsAdditionalData)
//	{
//		for(int x=0; x<10; x++)
//		{
//			int infoValue = sceneData[x+imageSize+imageSize];
//			for(int xx=0; xx<blockSize; xx++)
//			{
//				for(int yy=0; yy<blockSize; yy++)
//				{
//					uchar* ptr = map.ptr(yScreenSize*blockSize+yy)+(x*blockSize+xx)*3;
//					if(infoValue == 1) {ptr[0] = 255;ptr[1] = 255;ptr[2] = 255;}
//					else {ptr[0] = 0;ptr[1] = 0;ptr[2] = 0;}
//				}
//			}
//		}
//	}

	//Print
	imshow("AnalyzedSceneData", map);
	cv::waitKey(10);

}
