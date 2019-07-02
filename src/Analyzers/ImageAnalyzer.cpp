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
}

/*
 *
 */
ImageAnalyzer::AnalyzeResult ImageAnalyzer::processImage(cv::Mat* rawImage, cv::Mat* colorImage)
{
	ImageAnalyzer::AnalyzeResult analyzeResult;

	cv::Mat smallerImage = cutFragment(colorImage,cv::Point(0,64),cv::Point(512,448));

	//Player death
	analyzeResult.playerIsDead = false;
	analyzeResult.killedByEnemy = false;
	analyzeResult.playerWon = false;
	if(!findObject(smallerImage,deadImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,496,400)).empty())
	{	//Killed by Enemy
		analyzeResult.playerIsDead = true;
		analyzeResult.killedByEnemy = true;
	}
	if(!findObject(smallerImage,winImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,496,400)).empty())
	{	//Win
		analyzeResult.playerWon = true;
	}

	cv::Mat historyImage;
	if(oldImages.size() >3)
	{
		historyImage = getFirst(2, &(*oldImages.begin()));
		oldImages.erase(oldImages.begin());
	}
	else historyImage = cv::Mat(16, 16, CV_8UC3);


	reduceColors(0b10000000,&historyImage);

	int blockSize = 16;
	reduceColors(0b11000000,&smallerImage);
	cv::Mat firstPhaseImage = getMostFrequentInBlock(8, &smallerImage);
	cv::Mat processedImage = getLeastFrequentInImage(2, &firstPhaseImage);
	oldImages.push_back(processedImage.clone());

	viewImage(16,"proc2", processedImage);
	viewImage(32,"proc3", historyImage);

	analyzeResult.playerFound = true;
	analyzeResult.processedImagePast = historyImage;
	analyzeResult.processedImage = processedImage;
	return analyzeResult;
}

cv::Mat ImageAnalyzer::getMostFrequentInBlock(int blockSize, cv::Mat* image)
{
	cv::Mat processedImage = cv::Mat((image->rows)/blockSize, (image->cols)/blockSize, CV_8UC3);

	long long colors[64];

	for(int x=0,xb=0; x<image->cols-blockSize+1; xb++,x+=blockSize)
	{
		for(int y=0,yb=0; y<image->rows-blockSize+1; yb++,y+=blockSize)
		{
			for(int i=0; i<64; i++) colors[i] = -1;
			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptrSrc = image->ptr(y+yy)+(3*(x+xx));
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

			uchar* ptrDst = processedImage.ptr(yb)+((xb+xb+xb));

			ptrDst[0] = (pickedColor & 3 ) << 6;
			ptrDst[1] = (pickedColor & 12) << 4;
			ptrDst[2] = (pickedColor & 48) << 2;
		}
	}
	return processedImage;
}
cv::Mat ImageAnalyzer::getLeastFrequentInImage(int blockSize, cv::Mat* image)
{
	cv::Mat processedImage = cv::Mat((image->rows)/blockSize, (image->cols)/blockSize, CV_8UC3);

	long long colors[64];

	for(int i=0; i<64; i++) colors[i] = -1;
	for(int x=0; x<image->cols; x++)
	{
		for(int y=0; y<image->rows; y++)
		{
			uchar* ptrSrc = image->ptr(y)+(3*(x));
			int index = 0;
			index += ptrSrc[0] >> 6;
			index += ptrSrc[1] >> 4;
			index += ptrSrc[2] >> 2;
			colors[index] = colors[index] + 1;
		}
	}

	for(int x=0,xb=0; x<image->cols-blockSize+1; xb++,x+=blockSize)
	{
		for(int y=0,yb=0; y<image->rows-blockSize+1; yb++,y+=blockSize)
		{
			int pickedColor = 7;
			long long lowestCount = -1;

			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptrSrc = image->ptr(y+yy)+(3*(x+xx));
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

			uchar* ptrDst = processedImage.ptr(yb)+((xb+xb+xb));
			ptrDst[0] = (pickedColor & 3 ) << 6;
			ptrDst[1] = (pickedColor & 12) << 4;
			ptrDst[2] = (pickedColor & 48) << 2;
		}
	}

	return processedImage;
}

void ImageAnalyzer::reduceColors(int mask, cv::Mat* colorImage)
{
	for(int x=0; x<colorImage->cols; x++)
	{
		for(int y=0; y<colorImage->rows; y++)
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

	for(int x=0,xb=0; x<image->cols-blockSize+1; xb++,x+=blockSize)
	{
		for(int y=0,yb=0; y<image->rows-blockSize+1; yb++,y+=blockSize)
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

	for(int x=0; x<size.x; x++)
	{
		for(int y=0; y<size.y; y++)
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
			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptr = viewImage.ptr(y*blockSize+yy)+(x*blockSize+xx)*3;
					int v = (ptrSrc[0]+ptrSrc[1]+ptrSrc[2])/3;
					ptr[0] = 64*((int)ptrSrc[0]/64);
					ptr[1] = 64*((int)ptrSrc[1]/64);
					ptr[2] = 64*((int)ptrSrc[2]/64);
				}
			}
		}
	}
	//Print
	imshow(name, viewImage);
	cv::waitKey(10);
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
