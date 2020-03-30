/*
 * DataPrinter.cpp
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#include "DataDrawer.h"

/*
 *
 */
void DataDrawer::drawAnalyzedData(StateAnalyzer::AnalyzeResult& sceneData, ControllerInput t_keys, double reward, double change)
{
	int blockSize = 15;

	int xScreenSize = sceneData.fieldAndEnemiesLayout.cols;
	int yScreenSize = sceneData.fieldAndEnemiesLayout.rows;

	if(xScreenSize <= 0 && yScreenSize <=0) return;

	//map
	cv::Mat map = cv::Mat(blockSize*(yScreenSize+1), blockSize*(xScreenSize), CV_8UC3);
	for(int x=0, xb=0; x<xScreenSize; x++,xb+=blockSize)
	{
		for(int y=0, yb=0; y<yScreenSize; y++,yb+=blockSize)
		{
			cv::Scalar color;
			uchar* ptrSrc = sceneData.fieldAndEnemiesLayout.ptr(y)+(x+x+x);

			if(ptrSrc[0]==100) color = cv::Scalar(100,100,100); //field
			else if(ptrSrc[2]==220) color = cv::Scalar(0,0,220); //enemy
			else cv::Scalar(0,0,0); // blank

			if(x >> 1 == xScreenSize >> 2 && y >> 1 == yScreenSize >> 2)
			{
				color[0] += 50;
				color[1] += 50;
				color[2] += 50;
			}

			drawBlock(&map,blockSize,Point(xb,yb),color);
		}
	}
	for(int x=0; x<map.cols; x++)
	{
		for(int y=yScreenSize*blockSize; y<map.rows; y++)
		{
			uchar* ptr = map.ptr(y)+(x)*3;
			ptr[0] = ptr[1] = ptr[2] = 20;
		}
	}
	//player pos x
	drawBorderedBlock(&map,blockSize,
			Point(1*blockSize,yScreenSize*blockSize),
			cv::Scalar(sceneData.playerCoords.x,sceneData.playerCoords.x,sceneData.playerCoords.x));
	//player pos y
	drawBorderedBlock(&map,blockSize,
			Point(2*blockSize,yScreenSize*blockSize),
			cv::Scalar(sceneData.playerCoords.y,sceneData.playerCoords.y,sceneData.playerCoords.y));
	//player vel x
	drawBorderedBlock(&map,blockSize,
			Point(3*blockSize,yScreenSize*blockSize),
			cv::Scalar(
					0,
					sceneData.playerVelocity.x < 0 ? 0 : sceneData.playerVelocity.x*32,
					sceneData.playerVelocity.x > 0 ? 0 : sceneData.playerVelocity.x*32));
	//player vel y
	drawBorderedBlock(&map,blockSize,
			Point(4*blockSize,yScreenSize*blockSize),
			cv::Scalar(
					0,
					sceneData.playerVelocity.y < 0 ? 0 : sceneData.playerVelocity.y*32,
					sceneData.playerVelocity.y > 0 ? 0 : sceneData.playerVelocity.y*32));
	//pressed keys
	for(int i=0; i<t_keys.size(); i++)
	{
		int value = t_keys[i]>0 ? 255 : 0;
		drawBorderedBlock(&map,blockSize,
				Point((8+i)*blockSize,yScreenSize*blockSize),
				cv::Scalar(value,value,value));
	}

	reward = reward > 255 ? 255 : reward;
	reward = reward < -255 ? -255 : reward;
	//reward
	drawBorderedBlock(&map,blockSize,
			Point(20*blockSize,yScreenSize*blockSize),
			cv::Scalar(
					0,
					reward > 0 ? 5*reward : 0,
					reward < 0 ?   -reward : 0));
	//change
	change = abs(change);
	drawBorderedBlock(&map,blockSize,
			Point(21*blockSize,yScreenSize*blockSize),
			cv::Scalar(
					(change >= 40 && change < 9999) ? 255 : 0,
					change < 40 ? 255 : 0,
					change == 9999 ? 255 : 0));

	//Print
	imshow("AnalyzedSceneData", map);
	cv::waitKey(10);

}

/*
 *
 */
void DataDrawer::drawBlock(cv::Mat *mat, int t_blockSize, Point t_point, cv::Scalar t_color)
{
	for(int xx=0; xx<t_blockSize; xx++)
	{
		for(int yy=0; yy<t_blockSize; yy++)
		{
			uchar* ptr = mat->ptr(t_point.y+yy)+(t_point.x+xx)*3;
			ptr[0] = t_color[0];
			ptr[1] = t_color[1];
			ptr[2] = t_color[2];
		}
	}
}

/*
 *
 */
void DataDrawer::drawBorderedBlock(cv::Mat *mat, int t_blockSize, Point t_point, cv::Scalar t_color)
{
	for(int xx=0; xx<t_blockSize; xx++)
	{
		for(int yy=0; yy<t_blockSize; yy++)
		{
			uchar* ptr = mat->ptr(t_point.y+yy)+(t_point.x+xx)*3;
			ptr[0] = t_color[0];
			ptr[1] = t_color[1];
			ptr[2] = t_color[2];
			if(xx == 0 || yy == 0 || xx == t_blockSize-1 || yy == t_blockSize-1) {ptr[0] = ptr[1] = ptr[2] = 128;}
		}
	}
}
