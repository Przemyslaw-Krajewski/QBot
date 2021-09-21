/*
 * ImageAnalyzer.cpp
 *
 *  Created on: 15 wrz 2020
 *      Author: przemo
 */

#include "ImageAnalyzer.h"

/*
 *
 */
ImageAnalyzer::ImageAnalyzer(Game t_game)
{
	imageSize = cv::Point(256,256);
	game = t_game;
}

/*
 *
 */
ImageAnalyzer::~ImageAnalyzer()
{

}

/*
 *
 */
void ImageAnalyzer::correctScenarioHistory(std::list<SARS> &t_history, ScenarioAdditionalInfo t_additionalInfo)
{

}

/*
 *
 */
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
std::vector<cv::Point> ImageAnalyzer::findObject(cv::Mat &image, cv::Mat &pattern, cv::Point offset, cv::Scalar sample,
		                                         cv::Rect searchBounds)
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

