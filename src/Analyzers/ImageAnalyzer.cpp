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
}

/*
 *
 */
ImageAnalyzer::AnalyzeResult ImageAnalyzer::processImage(cv::Mat* image)
{
	ImageAnalyzer::AnalyzeResult analyzeResult;

	//Find objects
	cv::Point playerVelocity = cv::Point(0,0);
	cv::Point playerCoords = findPlayer(*image);

	std::vector<cv::Point> goombas,koopas,floors,walls,blocks1,blocks2,blocks3,pipe;
	if(image->ptr(1)[3]!=0 && image->ptr(0)[4]!=0 && image->ptr(0)[5]!=0)
	{
		//outside
		blocks1 = findObject(*image,blockImage1,		cv::Point(6,2),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks2	= findObject(*image,blockImage2,		cv::Point(0,0),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks3	= findObject(*image,blockImage3,		cv::Point(0,2),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		goombas	= findObject(*image,enemyImage1,		cv::Point(6,8),	cv::Scalar(0,0,0));
		koopas	= findObject(*image,enemyImage2,		cv::Point(4,0),	cv::Scalar(0,168,0));
		floors	= findObject(*image,floorImage1,		cv::Point(0,0),	cv::Scalar(176,188,252),cv::Rect(0,400,496,432));
		walls	= findObject(*image,wallimage1,		cv::Point(0,0),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		pipe	= findObject(*image,pipeImage,		cv::Point(0,0),	cv::Scalar(16,208,128),	cv::Rect(0,0,496,400));
	}
	else
	{
		//dungeon
		blocks1 = findObject(*image,blockImage1v2,	cv::Point(6,2),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks2	= findObject(*image,blockImage2v2,	cv::Point(0,0),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks3	= findObject(*image,blockImage3v2,	cv::Point(0,2),	cv::Scalar(136,128,0),	cv::Rect(0,0,496,400));
		goombas	= findObject(*image,enemyImage1v2,	cv::Point(6,8),	cv::Scalar(92,60,24));
		koopas	= findObject(*image,enemyImage2v2,	cv::Point(4,0),	cv::Scalar(136,128,0));
		floors	= findObject(*image,floorImage1v2,	cv::Point(0,0),	cv::Scalar(240,252,156),cv::Rect(0,400,496,432));
		walls	= findObject(*image,wallImage1v2,	cv::Point(0,0),	cv::Scalar(136,128,0),	cv::Rect(0,0,496,400));
		pipe	= findObject(*image,pipeImage,		cv::Point(0,0),	cv::Scalar(16,208,128),	cv::Rect(0,0,496,400));
	}

	//Player not found
	if(playerCoords == cv::Point(-1,-1))
	{
		analyzeResult.playerFound = false;
		return analyzeResult;
	}

	//clear image
	cv::Mat analyzedImage = cv::Mat(896, 512, CV_8UC3);
	for(int y = 0 ; y < analyzedImage.rows ; y++)
	{
		uchar* ptr = analyzedImage.ptr((int)y);
		for(int x = 0 ; x < analyzedImage.cols*3 ; x++)
		{
			*ptr=0;
			ptr = ptr+1;
		}
	}

	//Mark objects
	int interval = 16;
	cv::Point transl = cv::Point(256-playerCoords.x,448-playerCoords.y); // translation
	cv::Point blockSize = cv::Point(32,32);
	if(floors.size()>0)	rectangle(analyzedImage, cv::Rect(cv::Point(transl.x+blockSize.x, floors[0].y+transl.y),
											     	 	  cv::Point(895,                  floors[0].y+transl.y+blockSize.y)),
														  cv::Scalar(0,0,220), -1);
	for(cv::Point p : goombas) markObjectInImage(analyzedImage, blockSize, p, transl, cv::Point(0,0), 1);
	for(cv::Point p : koopas)  markObjectInImage(analyzedImage, blockSize, p, transl, cv::Point(0,0), 1);
	for(cv::Point p : floors)  markObjectInImage(analyzedImage, blockSize, p, transl, cv::Point(0,0), 0);
	for(cv::Point p : walls)   markObjectInImage(analyzedImage, blockSize, p, transl, cv::Point(0,0), 0);
	for(cv::Point p : blocks1) markObjectInImage(analyzedImage, blockSize, p, transl, cv::Point(-2,0), 0);
	for(cv::Point p : blocks2) markObjectInImage(analyzedImage, blockSize, p, transl, cv::Point(-2,2), 0);
	for(cv::Point p : blocks3) markObjectInImage(analyzedImage, blockSize, p, transl, cv::Point(0,2), 0);
	for(cv::Point p : pipe) rectangle(analyzedImage, cv::Rect(p+transl,cv::Point(p.x+64,416)+transl),cv::Scalar(100  ,100  ,100), -1);

	//Writing DATA OUTPUT
	cv::Mat fael = cv::Mat(imageSize.y*2/interval, imageSize.x/interval, CV_8UC3);
	for(int y = 0 ; y < fael.rows ; y++)
	{
		for(int x = 0 ; x < fael.cols ; x++)
		{
			uchar* ptrDst = fael.ptr(y)+x*3;
			uchar* ptrSrc = analyzedImage.ptr(y*interval+interval/2)+x*3*interval;
			ptrDst[0]=ptrSrc[0];ptrDst[1]=ptrSrc[1];ptrDst[2]=ptrSrc[2];
		}
	}

	//Player death
	bool playerIsDead = false;
	bool killedByEnemy = false;
	if(!findObject(*image,deadImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,496,400)).empty()) {playerIsDead = true; killedByEnemy = true;} //Killed by enemy
	if(playerCoords.y > 388) 																			playerIsDead = true; //pitfall
	if(playerCoords.x < 20) 																			playerIsDead = true; // left border

	#ifdef PRINT_ANALYZED_IMAGE
		//Print
		imshow("Objects", analyzedImage);
		cv::waitKey(10);
	#endif

	analyzeResult.fieldAndEnemiesLayout = fael;
	analyzeResult.playerIsDead = playerIsDead;
	analyzeResult.killedByEnemy = killedByEnemy;
	analyzeResult.playerFound = true;

	return analyzeResult;
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
