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
	reducedStateMethod = [](State & t_state) -> State
		{
			int xSize = 64;
			int ySize = 40;
			int zSize = 3;
			int imageOffset = 12;

			State median = State(0,xSize,ySize,zSize);
			State edges = State(0,xSize,ySize,1);
			State dilatation = State(0,xSize,ySize,1);

#ifdef PRINT_REDUCED_IMAGE
			DataDrawer::drawState(t_state,"original");
#endif
			//median filter
			for(int x=1; x<xSize-1; x++)
			{
				for(int y=1; y<ySize-1; y++)
				{
					int value = 0;
					cv::Scalar color;
					std::vector<cv::Scalar> v;
					for(int i=-1;i<=1;i++)
					{
						for(int j=-1;j<=1;j++)
						{
							int val = t_state[imageOffset + x+i + (y+j) *xSize] +
									  t_state[imageOffset + x+i + (y+j) *xSize + ySize*xSize] +
									  t_state[imageOffset + x+i + (y+j) *xSize + 2*ySize*xSize];
							if(val > value)
							{
								color = cv::Scalar(t_state[imageOffset + x+i + (y+j) *xSize],
										  	  	   t_state[imageOffset + x+i + (y+j) *xSize + ySize*xSize],
												   t_state[imageOffset + x+i + (y+j) *xSize + 2*ySize*xSize]);
								value = val;
							}
						}
					}
					median[x+ y*xSize] = 				 color.val[0];
					median[x+ y*xSize + xSize*ySize] =   color.val[1];
					median[x+ y*xSize + 2*xSize*ySize] = color.val[2];
				}
			}
#ifdef PRINT_REDUCED_IMAGE
			DataDrawer::drawState(median,"median");
#endif

			//Find edges and threshold
			for(int x=1; x<xSize-1; x++)
			{
				for(int y=1; y<ySize-1; y++)
				{
					int value = 0;
					for(int z=0; z<3; z++)
					{
						value += abs((median[ x +     y   *xSize + z*ySize*xSize])*4 -
									 (median[(x-1) +  y   *xSize + z*ySize*xSize]) -
									 (median[(x+1) +  y   *xSize + z*ySize*xSize]) -
									 (median[ x    + (y-1)*xSize + z*ySize*xSize]) -
									 (median[ x    + (y+1)*xSize + z*ySize*xSize]));
					}
					if(value > 120) edges[x+y*xSize] = 255;
					else edges[x+y*xSize] = 0;
				}
			}

#ifdef PRINT_REDUCED_IMAGE
			DataDrawer::drawState(edges,"edges");
#endif

			//Dilatation
			for(int x=2; x<xSize-2; x++)
			{
				for(int y=2; y<ySize-2; y++)
				{
					int value = 0;
					value = edges[x-1+(y-1)*xSize] + edges[x+(y-1)*xSize] + edges[x+1+(y-1)*xSize] +
							edges[x-1+y*xSize]     + edges[x+ y*xSize]    + edges[x+1+y*xSize] +
							edges[x-1+(y+1)*xSize] + edges[x+(y+1)*xSize] + edges[x+1+(y+1)*xSize];
					if(value >= 255*3) dilatation[x+y*xSize] = 255;
					else dilatation[x+y*xSize] = 0;
				}
			}

#ifdef PRINT_REDUCED_IMAGE
			DataDrawer::drawState(dilatation,"dilatation");
#endif

			//Reduce
			int reduceLevel = 4;
			int i = 0;
			State reduced = State(0,xSize/4,ySize/4,1);
			for(int y=0;y<ySize;y+=reduceLevel)
			{
				for(int x=0;x<xSize;x+=reduceLevel)
				{
					int value=0;
					for(int xx=0;xx<reduceLevel;xx++)
					{
						for(int yy=0;yy<reduceLevel;yy++)
						{
							value += dilatation[x+xx+(y+yy)*xSize];
						}
					}
					if(value > reduceLevel*64-1) reduced[i] = 255;
					else reduced[i] = 0;
					i++;
				}
			}

			xSize /= reduceLevel;
			ySize /= reduceLevel;

#ifdef PRINT_REDUCED_IMAGE
			DataDrawer::drawState(reduced,"reduced",32);
#endif

			//Reduce
			reduceLevel = 2;
			i = 0;
			State result = State(3,xSize/2,ySize/2,1);

			result[i] = t_state[0] == 0 ? 0 : (t_state[0] > 0 ? 1 : -1); i++;
			result[i] = t_state[1] == 0 ? 0 : (t_state[1] > 0 ? 1 : -1); i++;
			result[i] = t_state[3]; i++;

			for(int y=0;y<ySize;y+=reduceLevel)
			{
				for(int x=0;x<xSize;x+=reduceLevel)
				{
					int value=0;
					for(int xx=0;xx<reduceLevel;xx++)
					{
						for(int yy=0;yy<reduceLevel;yy++)
						{
							value += reduced[x+xx+(y+yy)*xSize];
						}
					}
					result[i] = (value/4);
					i++;
				}
			}

#ifdef PRINT_REDUCED_IMAGE
			DataDrawer::drawState(result,"result",64);
#endif

			return result;
		};

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
	//Calc sistuation
	if(game == Game::BattleToads) calculateSituationBT(colorImage,result);
	else if(game == Game::SuperMarioBros) calculateSituationSMB(colorImage,result);
	else throw std::string("RawImageAnalyzer::Not such game");
	result->playerFound = true;

	//Process Image
	result->processedImages.clear();

	cv::Mat cutImage = cutFragment(colorImage,cv::Point(0,64),cv::Point(256,224));
	reduceColors(0b11000000,&cutImage);

	cv::Mat firstPhaseImage, image;
	getLeastFrequentInImage(2, cutImage, firstPhaseImage);
	cv::resize(firstPhaseImage, image, cv::Size(), 0.5, 0.5,cv::INTER_CUBIC);
	oldImages.push_back(image.clone());
	result->processedImages.push_back(image);

	//Get past image
	if(oldImages.size() >=2)
	{
		result->processedImages.push_back(oldImages.begin()->clone());
		oldImages.erase(oldImages.begin());
	}
	else result->processedImages.push_back(cv::Mat(40, 64, CV_8UC3));

	//draw result
	viewImage(8,"processed image", result->processedImages[0]);
	viewImage(8,"image 2 frames ago", result->processedImages[1]);
}

/*
 *
 */
State RawImageAnalyzer::createSceneState(std::vector<cv::Mat> &t_images, ControllerInput& t_controllerInput, Point& t_position, Point& t_velocity)
{
	State sceneState(12, t_images[0].cols, t_images[0].rows, t_images[0].channels()*t_images.size());
	int i=0;

	//AdditionalInfo
	sceneState[i] = t_velocity.x; i++;
	sceneState[i] = t_velocity.y; i++;

	if(t_velocity.y == 0 && t_controllerInput[0] && holdButtonCounter <=1024) holdButtonCounter++;
	else holdButtonCounter = 0;

	sceneState[i] = t_velocity.y == 0; i++;
	sceneState[i] = holdButtonCounter >= 3 ? 64 : -64; i++;

	//Controller Info
	for(bool ci : t_controllerInput)
	{
		if(ci==true) sceneState[i] = MAX_INPUT_VALUE;
		else sceneState[i] = MIN_INPUT_VALUE;
		i++;
	}

	for(int j=0;j<2;j++) {sceneState[i] = 0; i++;}

	//Image info
	for(cv::Mat image : t_images)
	{
		for(int z=0; z<image.channels(); z++)
		{
			for(int y=0; y<image.rows; y++)
			{
				for(int x=0; x<image.cols; x++)
				{
					uchar* ptrSrc = image.ptr(y)+(3*(x));
					sceneState[i] = ptrSrc[z]; i++;
				}
			}
		}
	}

	return sceneState;
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
	if(!findObject(*image,winImage,cv::Point(10,5),cv::Scalar(0,148,0),cv::Rect(0,0,250,250)).empty())
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
