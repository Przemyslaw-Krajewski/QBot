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
void DataDrawer::drawState(State t_state, StateInfo t_stateInfo, std::string name)
{
	int blockSize = 8;

	cv::Mat mat = cv::Mat(blockSize*t_stateInfo.ySize, blockSize*t_stateInfo.xSize, CV_8UC3);

	for(int x=0, xb=0; x<t_stateInfo.xSize; x++,xb+=blockSize)
	{
		for(int y=0, yb=0; y<t_stateInfo.ySize; y++,yb+=blockSize)
		{
			cv::Scalar color;
			if(t_stateInfo.zSize == 1)
			{
				color[0] = color[1] = color[2] = t_state[x+y*t_stateInfo.xSize];
			}
			else if(t_stateInfo.zSize == 2)
			{
				color[1] = 0;
				color[0] = t_state[x+y*t_stateInfo.xSize];
				color[2] = t_state[x+y*t_stateInfo.xSize + t_stateInfo.xSize*t_stateInfo.ySize];
			}
			else if(t_stateInfo.zSize >= 3)
			{

				color[0] = t_state[x+y*t_stateInfo.xSize];
				color[1] = t_state[x+y*t_stateInfo.xSize + t_stateInfo.xSize*t_stateInfo.ySize];
				color[2] = t_state[x+y*t_stateInfo.xSize + t_stateInfo.xSize*t_stateInfo.ySize*2];
			}

			drawBlock(&mat,blockSize,Point(xb,yb),color);
		}
	}

	//Print
	imshow(name, mat);
	cv::waitKey(10);
}

/*
 *
 */
void DataDrawer::drawReducedState(State t_reducedState)
{
	int reduce = 8;
	int blockSize = 8*reduce;
	int xSize = 64/reduce;
	int ySize = 40/reduce;

	cv::Mat viewImage = cv::Mat(ySize*blockSize, xSize*blockSize, CV_8UC3);
	for(int x=0; x<xSize; x++)
	{
		for(int y=0; y<ySize; y++)
		{
			cv::Scalar color;
			int ptrSrc = t_reducedState[x*ySize+y];
			for(int yy=0; yy<blockSize; yy++)
			{
				for(int xx=0; xx<blockSize; xx++)
				{
					uchar* ptr = viewImage.ptr(y*blockSize+yy)+(x*blockSize+xx)*3;
					ptr[0] = ptrSrc;
					ptr[1] = ptrSrc;
					ptr[2] = ptrSrc;
				}
			}
		}
	}
	//Print
	imshow("ReducedState", viewImage);
	cv::waitKey(1);
}

void DataDrawer::drawAdditionalInfo(double t_reward, double t_maxTime, double t_time, ControllerInput t_keys, bool pressedKey)
{
	int blockSize = 15;
	int barWidth = 400;
	cv::Mat map = cv::Mat(blockSize*8, blockSize*2+barWidth, CV_8UC3);
	for(int x=0; x<map.cols; x++)
	{
		for(int y=0; y<map.rows; y++)
		{
			uchar* ptr = map.ptr(y)+(x)*3;
			ptr[0] = 0;
			ptr[1] = 0;
			ptr[2] = 0;
		}
	}

	drawBar(&map, blockSize, barWidth, (double) t_time/t_maxTime, cv::Point(blockSize,blockSize),
			cv::Scalar(0,255*t_time/t_maxTime,255*(1-t_time/t_maxTime)));

	double reward = t_reward/0.07;
	if(reward > 1) reward = 1;
	drawBar(&map, blockSize, barWidth, reward, cv::Point(blockSize,blockSize*3),cv::Scalar(255,255,255));

	//pressed keys
	for(int i=0; i<t_keys.size(); i++)
	{
		int value = t_keys[i] ? 255 : 0;
		drawBorderedBlock(&map,blockSize,
				Point((1+i)*blockSize,5*blockSize),
				cv::Scalar(value,value,value));
	}
	drawBorderedBlock(&map,blockSize,
			Point((1)*blockSize,7*blockSize),
			cv::Scalar(pressedKey*255,pressedKey*255,pressedKey*255));

	//Print
	imshow("Additional Info", map);
	cv::waitKey(10);
}

/*
 *
 */
void DataDrawer::drawBar(cv::Mat *mat, int t_barHeight, int t_barWidth, double progress, cv::Point t_point, cv::Scalar t_color)
{
	int progressBar = progress*t_barWidth;
	for(int xx=0; xx<t_barWidth; xx++)
	{
		for(int yy=0; yy<t_barHeight; yy++)
		{
			uchar* ptr = mat->ptr(t_point.y+yy)+(t_point.x+xx)*3;
			if(xx == 0 || yy == 0 || xx == t_barWidth-1 || yy == t_barHeight-1 )
			{
				ptr[0] = ptr[1] = ptr[2] = 255;
			}
			else if(xx < progressBar)
			{
				ptr[0] = t_color[0];
				ptr[1] = t_color[1];
				ptr[2] = t_color[2];
			}
		}
	}
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
