/*
 * RawImageAnalyzer.cpp
 *
 *  Created on: 5 wrz 2020
 *      Author: przemo
 */

#include "RawImageAnalyzer.h"

/*
 *
 */
RawImageAnalyzer::RawImageAnalyzer(Game t_game) : ImageAnalyzer(t_game)
{
	holdButtonCounter = 0;

	deadImage = cv::imread("graphics/dead.bmp", cv::IMREAD_COLOR);
	winImage = cv::imread("graphics/win.bmp", cv::IMREAD_COLOR);

	emptyHealth = cv::imread("graphics/EmptyHealth.bmp", cv::IMREAD_COLOR);
	hair = cv::imread("graphics/Hair.bmp", cv::IMREAD_COLOR);
}

/*
 *
 */
RawImageAnalyzer::~RawImageAnalyzer()
{

}

/*
 *
 */
void RawImageAnalyzer::processImage(cv::Mat* colorImage, ImageAnalyzer::AnalyzeResult *result)
{

	cv::Mat smallerImage = cutFragment(colorImage,cv::Point(0,64),cv::Point(256,224));

	if(game == Game::BattleToads) calculateSituationBT(colorImage,result);
	else if(game == Game::SuperMarioBros) calculateSituationSMB(colorImage,result);
	else throw std::string("RawImageAnalyzer::Not such game");

	if(oldImages.size() >2)
	{
		result->processedImagePast = *oldImages.begin();
		oldImages2.push_back(result->processedImagePast.clone());
		oldImages.erase(oldImages.begin());
	}
	else result->processedImagePast = cv::Mat(40, 64, CV_8UC3);

//	if(oldImages2.size() >6)
//	{
//		result->processedImagePast2 = *oldImages2.begin();
//		oldImages2.erase(oldImages2.begin());
//	}
//	else result->processedImagePast2 = cv::Mat(10, 16, CV_8UC3);
//	reduceColors(0b10000000,&(result->processedImagePast));

	int blockSize = 16;
	reduceColors(0b11000000,&smallerImage);
	cv::Mat firstPhaseImage;
//	getMostFrequentInBlock(2, smallerImage, firstPhaseImage);
	getLeastFrequentInImage(2, smallerImage, firstPhaseImage);
//	cv::medianBlur(	smallerImage,smallerImage,3);
	cv::resize(firstPhaseImage, result->processedImage, cv::Size(), 0.5, 0.5,CV_INTER_CUBIC);
	oldImages.push_back(result->processedImage);

	viewImage(8,"proc", result->processedImage);
	viewImage(8,"past1", result->processedImagePast);
//	viewImage(32,"past2", result->processedImagePast2);
	result->playerFound = true;
}

/*
 *
 */
std::vector<int> RawImageAnalyzer::createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
		ControllerInput& controllerInput, Point& position, Point& velocity)
{
	State sceneState;
	for(int z=0; z<image.channels(); z++)
	{
		for(int y=0; y<image.rows; y++)
		{
			for(int x=0; x<image.cols; x++)
			{
				uchar* ptrSrc = image.ptr(y)+(3*(x));
				sceneState.push_back(ptrSrc[z]);
			}
		}
	}

	for(int z=0; z<imagePast.channels(); z++)
	{
		for(int y=0; y<imagePast.rows; y++)
		{
			for(int x=0; x<imagePast.cols; x++)
			{
				uchar* ptrSrc = imagePast.ptr(y)+(3*(x));
				sceneState.push_back(ptrSrc[z]);
			}
		}
	}

//	for(int x=0; x<imagePast2.cols; x++)
//	{
//		for(int y=0; y<imagePast2.rows; y++)
//		{
//			uchar* ptrSrc = imagePast2.ptr(y)+(3*(x));
//			sceneState.push_back((ptrSrc[0] >> 7) + (ptrSrc[1] >> 6) + (ptrSrc[2] >> 5));
//		}
//	}

	//Controller
	for(bool ci : controllerInput)
	{
		if(ci==true) sceneState.push_back(MAX_INPUT_VALUE);
		else sceneState.push_back(MIN_INPUT_VALUE);
	}

	//AdditionalInfo
	sceneState.push_back(velocity.x/4);
	sceneState.push_back(velocity.y/2);

	if(velocity.y == 0 && controllerInput[0] && holdButtonCounter <=1024) holdButtonCounter++;
	else holdButtonCounter = 0;

	sceneState.push_back(velocity.y == 0);
	sceneState.push_back(holdButtonCounter >= 2 ? MAX_INPUT_VALUE : MIN_INPUT_VALUE);

	for(int i=0;i<6;i++) sceneState.push_back(0);

	return sceneState;
}

/*
 *
 */
State RawImageAnalyzer::reduceSceneState(const State& t_state, double action)
{
	int reduceLevel = 4;
	int xSize = 64;
	int ySize = 40;
	int zSize = 3;

	std::vector<int> edges = std::vector<int>(xSize*ySize,0);
	std::vector<int> erosion = std::vector<int>(xSize*ySize,0);

	//Find edges and threshold
	for(int x=1; x<xSize-1; x++)
	{
		for(int y=1; y<ySize-1; y++)
		{
			int value = 0;
			for(int z=0; z<zSize; z++)
			{
				value += abs((t_state[ x +     y   *xSize + z*ySize*xSize]>> 0)*4 -
							 (t_state[(x-1) +  y   *xSize + z*ySize*xSize]>> 0) -
							 (t_state[(x+1) +  y   *xSize + z*ySize*xSize]>> 0) -
							 (t_state[ x    + (y-1)*xSize + z*ySize*xSize]>> 0) -
							 (t_state[ x    + (y+1)*xSize + z*ySize*xSize]>> 0));
			}
			if(value > 0) edges[x+y*xSize] = 255;
//			else if(value > 20) edges[x+y*xSize] = 128;
//			else edges[x+y*xSize] = 0;
//			edges[x+y*xSize] = value;
		}
	}

	//Erosion
	for(int x=2; x<xSize-2; x++)
	{
		for(int y=2; y<ySize-2; y++)
		{
			int value = 0;
			value = edges[x-1+(y-1)*xSize] + edges[x+(y-1)*xSize] + edges[x+1+(y-1)*xSize] +
					edges[x-1+y*xSize] + 					+ edges[x+1+y*xSize] +
					edges[x-1+(y+1)*xSize] + edges[x+(y+1)*xSize] + edges[x+1+(y+1)*xSize];
//			value = edges[x-2+(y-2)*xSize] + edges[x-1+(y-2)*xSize] + edges[x+(y-2)*xSize] + edges[x+1+(y-2)*xSize] + edges[x+2+(y-2)*xSize] +
//								edges[x-2+(y-1)*xSize] + edges[x-1+(y-1)*xSize] + edges[x+(y-1)*xSize] + edges[x+1+(y-1)*xSize] + edges[x+2+(y-1)*xSize] +
//								edges[x-2+y*xSize] + edges[x-1+y*xSize] + 					+ edges[x+1+y*xSize] + edges[x+2+y*xSize] +
//								edges[x-2+(y+1)*xSize] + edges[x-1+(y+1)*xSize] + edges[x+(y+1)*xSize] + edges[x+1+(y+1)*xSize] + edges[x+2+(y+1)*xSize] +
//								edges[x-2+(y+2)*xSize] + edges[x-1+(y+2)*xSize] + edges[x+(y+2)*xSize] + edges[x+1+(y+2)*xSize] + edges[x+2+(y+2)*xSize];
			if(value >= 255*8) erosion[x+y*xSize] = 255;
		}
	}

	//Reduce
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
					value += erosion[x+xx+(y+yy)*xSize];
//					if(edges[x+xx+(y+yy)*xSize] > value) value = erosion[x+xx+(y+yy)*xSize];
				}
			}
			if(value > reduceLevel*64) result.push_back(255);
			else result.push_back(0);
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
	result.push_back(t_state[t_state.size()-4-6]/4);
	result.push_back(t_state[t_state.size()-3-6]/2);
	result.push_back(10*t_state[t_state.size()-1-6]);

//	result.push_back(action);

	return result;
}

/*
 *
 */
void RawImageAnalyzer::calculateSituationSMB(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult)
{
	//Player death
	analyzeResult->playerIsDead = false;
	analyzeResult->killedByEnemy = false;
	analyzeResult->playerWon = false;
	if(!findObject(*image,deadImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,250,250)).empty())
	{	//Killed by Enemy
		analyzeResult->playerIsDead = true;
		analyzeResult->killedByEnemy = true;
	}
	if(!findObject(*image,winImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,250,250)).empty())
	{	//Win
		analyzeResult->playerWon = true;
	}
}

/*
 *
 */
void RawImageAnalyzer::calculateSituationBT(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult)
{
	//Player death
	analyzeResult->playerIsDead = false;
	analyzeResult->playerWon = false;
	if(!findObject(*image,emptyHealth,cv::Point(0,0),cv::Scalar(96,116,252),cv::Rect(128,15,256,25)).empty())
	{
		analyzeResult->playerIsDead = true;
	}
	if(!findObject(*image,hair,cv::Point(0,0),cv::Scalar(168,0,0),cv::Rect(95,50,100,55)).empty())
	{
		analyzeResult->playerWon = true;
	}
}

/*
 *
 */
void RawImageAnalyzer::getMostFrequentInBlock(int blockSize, cv::Mat& srcImage, cv::Mat& dstImage)
{
	dstImage = cv::Mat((srcImage.rows)/blockSize, (srcImage.cols)/blockSize, CV_8UC3);

	long long colors[64];

	for(int y=0,yb=0; y<srcImage.rows-blockSize+1; yb++,y+=blockSize)
	{
		for(int x=0,xb=0; x<srcImage.cols-blockSize+1; xb++,x+=blockSize)
		{
			for(int i=0; i<64; i++) colors[i] = -1;
			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptrSrc = srcImage.ptr(y+yy)+(3*(x+xx));
					int index = 0;
					index += ptrSrc[0] >> 6;
					index += ptrSrc[1] >> 4;
					index += ptrSrc[2] >> 2;
					colors[index] = colors[index] + 1;
				}
			}

			int pickedColor = 7;
			long long highestCount = -1;
			for(int i=0; i<64; i++)
			{
				if((highestCount < colors[i] || highestCount == -1))
				{
					highestCount = colors[i];
					pickedColor = i;
				}
			}

			uchar* ptrDst = dstImage.ptr(yb)+((xb+xb+xb));

			ptrDst[0] = (pickedColor & 3 ) << 6;
			ptrDst[1] = (pickedColor & 12) << 4;
			ptrDst[2] = (pickedColor & 48) << 2;
		}
	}
}

/*
 *
 */
void RawImageAnalyzer::getLeastFrequentInImage(int blockSize, cv::Mat& srcImage, cv::Mat& dstImage)
{
	dstImage = cv::Mat((srcImage.rows)/blockSize, (srcImage.cols)/blockSize, CV_8UC3);

	long long colors[64];

	for(int i=0; i<64; i++) colors[i] = -1;
	for(int y=0; y<srcImage.rows; y++)
	{
		for(int x=0; x<srcImage.cols; x++)
		{
			uchar* ptrSrc = srcImage.ptr(y)+(3*(x));
			int index = 0;
			index += ptrSrc[0] >> 6;
			index += ptrSrc[1] >> 4;
			index += ptrSrc[2] >> 2;
			colors[index] = colors[index] + 1;
		}
	}

	for(int y=0,yb=0; y<srcImage.rows-blockSize+1; yb++,y+=blockSize)
	{
		for(int x=0,xb=0; x<srcImage.cols-blockSize+1; xb++,x+=blockSize)
		{
			int pickedColor = 7;
			long long lowestCount = -1;

			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptrSrc = srcImage.ptr(y+yy)+(3*(x+xx));
					int index = 0;
					index += ptrSrc[0] >> 6;
					index += ptrSrc[1] >> 4;
					index += ptrSrc[2] >> 2;

					if((lowestCount > colors[index] || lowestCount == -1))
					{
						lowestCount = colors[index];
						pickedColor = index;
					}
				}
			}

			uchar* ptrDst = dstImage.ptr(yb)+((xb+xb+xb));
			ptrDst[0] = (pickedColor & 3 ) << 6;
			ptrDst[1] = (pickedColor & 12) << 4;
			ptrDst[2] = (pickedColor & 48) << 2;
		}
	}
}

/*
 *
 */
void RawImageAnalyzer::reduceColors(int mask, cv::Mat* colorImage)
{
	for(int y=0; y<colorImage->rows; y++)
	{
		for(int x=0; x<colorImage->cols; x++)
		{
			uchar* ptr = colorImage->ptr(y)+((x+x+x));
			ptr[0] = ptr[0] & mask;
			ptr[1] = ptr[1] & mask;
			ptr[2] = ptr[2] & mask;
		}
	}
}

/*
 *
 */
cv::Mat RawImageAnalyzer::getFirst(int blockSize, cv::Mat* image)
{
	cv::Mat processedImage = cv::Mat((image->rows)/blockSize, (image->cols)/blockSize, CV_8UC3);

	for(int y=0,yb=0; y<image->rows-blockSize+1; yb++,y+=blockSize)
	{
		for(int x=0,xb=0; x<image->cols-blockSize+1; xb++,x+=blockSize)
		{
			uchar* ptrSrc = image->ptr(y)+(3*(x));
			uchar* ptrDst = processedImage.ptr(yb)+((xb+xb+xb));
			ptrDst[0] = ptrSrc[0];
			ptrDst[1] = ptrSrc[1];
			ptrDst[2] = ptrSrc[2];
		}
	}

	return processedImage;
}
