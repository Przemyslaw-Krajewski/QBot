/*
 * ReinforcementLearning.cpp
 *
 *  Created on: 16 wrz 2020
 *      Author: przemo
 */

#include "ReinforcementLearning.h"

ReinforcementLearning::ReinforcementLearning()
{

}

ReinforcementLearning::~ReinforcementLearning()
{

}

/*
 *
 */
int ReinforcementLearning::getIndexOfMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return maxIndex;
}

/*
 *
 */
double ReinforcementLearning::getMaxValue(std::vector<double> t_array)
{
	int maxIndex = 0;
	for(int i=1; i<t_array.size(); i++)
	{
		if(t_array[i] > t_array[maxIndex]) maxIndex = i;
	}
	return t_array[maxIndex];
}

/*
 *
 */
State ReinforcementLearning::reduceSceneState(const State& t_state, double action)
{
	//Metadata
//	int reduceLevel = 8;
//	int xSize = 32;
//	int ySize = 56;
//
//	std::vector<int> result;
//	for(int x=0;x<32;x+=reduceLevel)
//	{
//		for(int y=0;y<56;y+=reduceLevel)
//		{
//			int value=0;
//			for(int xx=0;xx<reduceLevel;xx++)
//			{
//				for(int yy=0;yy<reduceLevel;yy++)
//				{
//					if(t_state[(x+xx)*56+y+yy] > value) value = t_state[+(x+xx)*56+y+yy];
//					if(t_state[32*56+(x+xx)*56+y+yy]*2 > value) value = t_state[32*56+(x+xx)*56+y+yy]*2;
//				}
//			}
//			result.push_back(value);
//		}
//	}

	//New raw image find edges
	int reduceLevel = 4;
	int xSize = 64;
	int ySize = 40;
	int zSize = 3;

	std::vector<int> edges = std::vector<int>(xSize*ySize,0);

	//Find edges and threshold
	for(int x=1; x<xSize-1; x++)
	{
		for(int y=1; y<ySize-1; y++)
		{
			int value = 0;
			for(int z=0; z<zSize; z++)
			{
				value += abs((t_state[ x +     y   *xSize + z*ySize*xSize]>> 4)*4 -
							 (t_state[(x-1) +  y   *xSize + z*ySize*xSize]>> 4) -
							 (t_state[(x+1) +  y   *xSize + z*ySize*xSize]>> 4) -
							 (t_state[ x    + (y-1)*xSize + z*ySize*xSize]>> 4) -
							 (t_state[ x    + (y+1)*xSize + z*ySize*xSize]>> 4));
			}
			edges[x+y*xSize] = value > 30 ? 1 : 0;
		}
	}

	std::vector<int> result;
	for(int x=0;x<xSize;x+=reduceLevel)
	{
		for(int y=0;y<ySize;y+=reduceLevel)
		{
			int value=0;
			for(int xx=0;xx<reduceLevel;xx++)
			{
				for(int yy=0;yy<reduceLevel;yy++)
				{
					if(edges[x+xx+(y+yy)*xSize] > value) value = edges[x+xx+(y+yy)*xSize];
				}
			}
			result.push_back(value);
		}
	}

	//Old raw image
//	std::vector<int> result;
//	for(int x=0;x<xSize;x+=reduceLevel)
//	{
//		for(int y=0;y<ySize;y+=reduceLevel)
//		{
//			for(int z=0; z<zSize; z++)
//			{
//				int value = (t_state[x + y*xSize + z*ySize*xSize] >> 7);
//				result.push_back(value);
//			}
//		}
//	}
	result.push_back(t_state[t_state.size()-4]/3);
	result.push_back(t_state[t_state.size()-3]);
	result.push_back(t_state[t_state.size()-1]);

//	result.push_back(action);

	return result;
}

/*
 *
 */
void ReinforcementLearning::drawReducedSceneState(const State& t_state)
{
	int reduce = 4;
	int blockSize = 8*reduce;
	int xSize = 64/reduce;
	int ySize = 40/reduce;

	std::vector<int> result = reduceSceneState(t_state,0);

	cv::Mat viewImage = cv::Mat(ySize*blockSize, xSize*blockSize, CV_8UC3);
	for(int x=0; x<xSize; x++)
	{
		for(int y=0; y<ySize; y++)
		{
			cv::Scalar color;
			int ptrSrc = result[x*ySize+y] > 0 ? 255 : 0;
//			int ptrSrc = result[x*ySize+y]*16;
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
