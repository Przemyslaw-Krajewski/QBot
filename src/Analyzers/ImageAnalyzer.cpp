/*
 * Analizator.cpp
 *
 *  Created on: 3 gru 2017
 *      Author: przemo
 */

#include "../Analyzers/ImageAnalyzer.h"

/*
 *
 */
ImageAnalyzer::ImageAnalyzer()
{
	imageSize = cv::Point(512,448);

	cv::Mat image = cv::imread("graphics/mario.bmp", cv::IMREAD_COLOR);
	playerHistogram = determineHistogram(image);
	enemyImage1 = cv::imread("graphics/goomba.bmp", cv::IMREAD_COLOR);
	enemyImage1v2 = cv::imread("graphics/goomba2.bmp", cv::IMREAD_COLOR);

	enemyImage2 = cv::imread("graphics/koopa.bmp", cv::IMREAD_COLOR);
	enemyImage2v2 = cv::imread("graphics/koopa2.bmp", cv::IMREAD_COLOR);

	floorImage1 = cv::imread("graphics/floor.bmp", cv::IMREAD_COLOR);
	floorImage1v2 = cv::imread("graphics/floor2.bmp", cv::IMREAD_COLOR);

	wallimage1 = cv::imread("graphics/wall.bmp", cv::IMREAD_COLOR);
	wallImage1v2 = cv::imread("graphics/wall2.bmp", cv::IMREAD_COLOR);

	blockImage1 = cv::imread("graphics/block1.bmp", cv::IMREAD_COLOR);
	blockImage1v2 = cv::imread("graphics/block1v2.bmp", cv::IMREAD_COLOR);

	blockImage2 = cv::imread("graphics/block2.bmp", cv::IMREAD_COLOR);
	blockImage2v2 = cv::imread("graphics/block2v2.bmp", cv::IMREAD_COLOR);

	blockImage3 = cv::imread("graphics/block3.bmp", cv::IMREAD_COLOR);
	blockImage3v2 = cv::imread("graphics/block3v2.bmp", cv::IMREAD_COLOR);

	pipeImage = cv::imread("graphics/pipe.bmp", cv::IMREAD_COLOR);
	deadImage = cv::imread("graphics/dead.bmp", cv::IMREAD_COLOR);
	mushroomImage = cv::imread("graphics/mushroom.bmp", cv::IMREAD_COLOR);
	cloudImage = cv::imread("graphics/cloud.bmp", cv::IMREAD_COLOR);
	winImage = cv::imread("graphics/win.bmp", cv::IMREAD_COLOR);

	emptyHealth = cv::imread("graphics/EmptyHealth.bmp", cv::IMREAD_COLOR);
	hair = cv::imread("graphics/Hair.bmp", cv::IMREAD_COLOR);
}

/*
 *
 */
void ImageAnalyzer::processImage(cv::Mat* colorImage, ImageAnalyzer::AnalyzeResult *result)
{

	cv::Mat smallerImage = cutFragment(colorImage,cv::Point(0,64),cv::Point(256,224));

	calculateSituationBT(colorImage,result);

	if(oldImages.size() >6)
	{
		result->processedImagePast = *oldImages.begin();
		oldImages2.push_back(result->processedImagePast.clone());
		oldImages.erase(oldImages.begin());
	}
	else result->processedImagePast = cv::Mat(20, 32, CV_8UC3);

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
	getMostFrequentInBlock(2, smallerImage, firstPhaseImage);
	getLeastFrequentInImage(4, firstPhaseImage, result->processedImage);
	oldImages.push_back(result->processedImage);

	viewImage(16,"proc", result->processedImage);
	viewImage(16,"past1", result->processedImagePast);
//	viewImage(32,"past2", result->processedImagePast2);
	result->playerFound = true;
}

void ImageAnalyzer::calculateSituationSMB(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult)
{
	//Player death
	analyzeResult->playerIsDead = false;
	analyzeResult->killedByEnemy = false;
	analyzeResult->playerWon = false;
	if(!findObject(*image,deadImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,496,400)).empty())
	{	//Killed by Enemy
		analyzeResult->playerIsDead = true;
		analyzeResult->killedByEnemy = true;
	}
	if(!findObject(*image,winImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,496,400)).empty())
	{	//Win
		analyzeResult->playerWon = true;
	}
}

void ImageAnalyzer::calculateSituationBT(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult)
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

void ImageAnalyzer::getMostFrequentInBlock(int blockSize, cv::Mat& srcImage, cv::Mat& dstImage)
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
void ImageAnalyzer::getLeastFrequentInImage(int blockSize, cv::Mat& srcImage, cv::Mat& dstImage)
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

void ImageAnalyzer::reduceColors(int mask, cv::Mat* colorImage)
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

cv::Mat ImageAnalyzer::getFirst(int blockSize, cv::Mat* image)
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

std::vector<int> ImageAnalyzer::toVector(cv::Mat *image)
{
	std::vector<int> result;
	for(int x=0; x<image->cols; x++)
	{
		for(int y=0; y<image->rows; y++)
		{
			uchar* ptrSrc = image->ptr(y)+(3*(x));
			result.push_back((ptrSrc[0] >> 6) + (ptrSrc[1] >> 4) + (ptrSrc[2] >> 2));
		}
	}

	return result;
}

cv::Mat ImageAnalyzer::cutFragment(cv::Mat* image, cv::Point leftUp, cv::Point rightDown)
{
	cv::Point size = rightDown - leftUp;
	cv::Mat result = cv::Mat(size.y, size.x, CV_8UC3);

	for(int y=0; y<size.y; y++)
	{
		for(int x=0; x<size.x; x++)
		{
			uchar* ptrSrc = image->ptr(y+leftUp.y)+(3*(x+leftUp.x));
			uchar* ptrDst = result.ptr(y)+((x+x+x));
			ptrDst[0] = ptrSrc[0];
			ptrDst[1] = ptrSrc[1];
			ptrDst[2] = ptrSrc[2];
		}
	}

	return result;
}

void ImageAnalyzer::viewImage(int blockSize, std::string name, cv::Mat &image)
{
#ifdef PRINT_ANALYZED_IMAGE
	int mask = 0b11000000;
	//View
	int xScreenSize = image.cols;
	int yScreenSize = image.rows;
	cv::Mat viewImage = cv::Mat((image.rows)*blockSize, (image.cols)*blockSize, CV_8UC3);
	for(int x=0; x<xScreenSize; x++)
	{
		for(int y=0; y<yScreenSize; y++)
		{
			cv::Scalar color;
			uchar* ptrSrc = image.ptr(y)+(x+x+x);
			for(int yy=0; yy<blockSize; yy++)
			{
				for(int xx=0; xx<blockSize; xx++)
				{
					uchar* ptr = viewImage.ptr(y*blockSize+yy)+(x*blockSize+xx)*3;
					ptr[0] = ptrSrc[0];
					ptr[1] = ptrSrc[1];
					ptr[2] = ptrSrc[2];
				}
			}
		}
	}
	//Print
	imshow(name, viewImage);
	cv::waitKey(1);
#endif
}

/*
 *
 */
cv::Point ImageAnalyzer::findPlayer(cv::Mat &image)
{
	int channels = image.channels();
	int width = image.cols;
	int height = image.rows;
	for(int y=20 ; y<height-20 ; y++)
	{
		for(int x=20 ; x<width-20 ; x++)
		{
			uchar* ptr = image.ptr(y)+x*channels;
			if(ptr[0]==0 && ptr[1]==148 && ptr[2]==0)
			{
				cv::Mat pattern = copyMat(image,cv::Point(x-4,y-4),cv::Point(28,36));
				Histogram histogram = determineHistogram(pattern);
				double r1 = cv::compareHist( histogram[0], playerHistogram[0], 3 );//CV_COMP_INTERSECT
				double r2 = cv::compareHist( histogram[1], playerHistogram[1], 3 );//CV_COMP_INTERSECT
				double r3 = cv::compareHist( histogram[2], playerHistogram[2], 3 );//CV_COMP_INTERSECT
				if(r1<0.70 && r2<0.7 && r3<0.70)
				{
					return cv::Point(x-4,y-4);
				}
			}
		}
	}
	return cv::Point(-1,-1);
}

/*
 *
 */
ImageAnalyzer::Histogram ImageAnalyzer::determineHistogram(cv::Mat &image)
{
	std::vector<cv::Mat> bgr_planes;
	cv::split( image, bgr_planes );

	int histogramSize = 256;
	float range[] = { 0, 256 } ;
	const float* histogramRange = { range };
	std::vector<cv::Mat> results;
	results.push_back(cv::Mat());
	results.push_back(cv::Mat());
	results.push_back(cv::Mat());

	cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), results[0], 1, &histogramSize, &histogramRange);
	cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), results[1], 1, &histogramSize, &histogramRange);
	cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), results[2], 1, &histogramSize, &histogramRange);

	return results;
}

/*
 *
 */
std::vector<cv::Point> ImageAnalyzer::findObject(cv::Mat &image, cv::Mat &pattern, cv::Point offset, cv::Scalar sample, cv::Rect searchBounds)
{
	std::vector<cv::Point> result = std::vector<cv::Point>();
	int channels = image.channels();
	int width = image.cols;
	int height = image.rows;
	cv::Point size = cv::Point(pattern.cols,pattern.rows);
	if(searchBounds.x == -1)
	{
		searchBounds.x = 0;
		searchBounds.y = size.y/2;
		searchBounds.width = width-size.x/2;
		searchBounds.height = height-size.y/2;
	}
	for(int y=searchBounds.y ; y<searchBounds.height ; y++)
	{
		for(int x=searchBounds.x ; x<searchBounds.width ; x++)
		{
			uchar* ptr = image.ptr(y)+x*channels;
			if(ptr[0]==sample[0] && ptr[1]==sample[1] && ptr[2]==sample[2])
			{
				if(compareMat(image,pattern,cv::Point(x-offset.x,y-offset.y)))
				{
					result.push_back(cv::Point(x-offset.x,y-offset.y));
				}
			}
		}
	}
	return result;
}

/*
 *
 */
cv::Mat ImageAnalyzer::copyMat(cv::Mat src, cv::Point offset, cv::Point size)
{
	cv::Mat dst = cv::Mat(size.y,size.x,CV_8UC3);
	for(int y=0 ; y<size.y ; y++)
	{
		for(int x=0 ; x<size.x ; x++)
		{
			uchar* ptrs = src.ptr(y+offset.y)+(x+offset.x)*src.channels();
			uchar* ptrd = dst.ptr(y)+x*dst.channels();
			ptrd[0]=ptrs[0];ptrd[1]=ptrs[1];ptrd[2]=ptrs[2];
		}
	}
	return dst;
}

/*
 *
 */
bool ImageAnalyzer::compareMat(cv::Mat &image, cv::Mat &pattern, cv::Point offset)
{
	for(int y=0 ; y<pattern.rows ; y++)
	{
		uchar* ptr1 = pattern.ptr(y);
		uchar* ptr2 = image.ptr(y+offset.y)+(offset.x)*pattern.channels();
		for(int x=0 ; x<pattern.cols ; x++)
		{
			if(!(ptr1[0]==128 && ptr1[1]==0 && ptr1[2]==255) && !(ptr1[0]==ptr2[0] && ptr1[1]==ptr2[1] && ptr1[2]==ptr2[2])) return false;
			ptr1 += pattern.channels();
			ptr2 += pattern.channels();
		}
	}
	return true;
}

/*
 *
 */
bool ImageAnalyzer::compareMat(cv::Mat &mat1, cv::Mat &mat2)
{
	if(mat1.rows!=mat2.rows && mat1.cols!=mat2.cols) return false;
	for(int y=0 ; y<mat1.rows ; y++)
	{
		for(int x=0 ; x<mat1.cols ; x++)
		{
			uchar* ptr1 = mat1.ptr(y)+x*mat1.channels();
			uchar* ptr2 = mat2.ptr(y)+x*mat2.channels();
			if(!(ptr1[0]==128 && ptr1[1]==0 && ptr1[2]==255) && !(ptr1[0]==ptr2[0] && ptr1[1]==ptr2[1] && ptr1[2]==ptr2[2])) return false;
		}
	}
	return true;
}

void ImageAnalyzer::markObjectInImage(cv::Mat& resultImage, cv::Point blockSize, cv::Point point, cv::Point translation, cv::Point correction, int objectType)
{
	cv::Scalar color = cv::Scalar(0  ,0  ,220);
	if(objectType == 2) color = cv::Scalar(255 ,255 ,255);
	else if(objectType == 1) color = cv::Scalar(0 ,0 ,220);
	else if(objectType == 0) color = cv::Scalar(100 ,100 ,100);

	rectangle(resultImage, cv::Rect(point+translation+correction,
									point+translation+correction+blockSize),
									color, -1);
}
