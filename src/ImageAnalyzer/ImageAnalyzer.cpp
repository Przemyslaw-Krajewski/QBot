/*
 * Analizator.cpp
 *
 *  Created on: 3 gru 2017
 *      Author: przemo
 */

#include "../ImageAnalyzer/ImageAnalyzer.h"

/*
 *
 */
ImageAnalyzer::ImageAnalyzer()
{
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
ImageAnalyzer::AnalyzeResult ImageAnalyzer::processImage()
{
	ImageAnalyzer::AnalyzeResult analyzeResult;

	//Get Desktop Screen
	cv::Mat image;
	image = cv::Mat(448, 512, CV_8UC3);
	DesktopHandler::getPtr()->getDesktop(&image);

	//Find objects
	cv::Point playerVelocity = cv::Point(0,0);
	cv::Point playerCoords = findPlayer(image);

	std::vector<cv::Point> goombas,koopas,floors,walls,blocks1,blocks2,blocks3,pipe;
	if(image.ptr(1)[3]!=0 && image.ptr(0)[4]!=0 && image.ptr(0)[5]!=0)
	{
		//outside
		blocks1 = findObject(image,blockImage1,		cv::Point(6,2),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks2	= findObject(image,blockImage2,		cv::Point(0,0),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks3	= findObject(image,blockImage3,		cv::Point(0,2),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		goombas	= findObject(image,enemyImage1,		cv::Point(6,8),	cv::Scalar(0,0,0));
		koopas	= findObject(image,enemyImage2,		cv::Point(4,0),	cv::Scalar(0,168,0));
		floors	= findObject(image,floorImage1,		cv::Point(0,0),	cv::Scalar(176,188,252),cv::Rect(0,400,496,432));
		walls	= findObject(image,wallimage1,		cv::Point(0,0),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		pipe	= findObject(image,pipeImage,		cv::Point(0,0),	cv::Scalar(16,208,128),	cv::Rect(0,0,496,400));
	}
	else
	{
		//dungeon
		blocks1 = findObject(image,blockImage1v2,	cv::Point(6,2),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks2	= findObject(image,blockImage2v2,	cv::Point(0,0),	cv::Scalar(12,76,200),	cv::Rect(0,0,496,400));
		blocks3	= findObject(image,blockImage3v2,	cv::Point(0,2),	cv::Scalar(136,128,0),	cv::Rect(0,0,496,400));
		goombas	= findObject(image,enemyImage1v2,	cv::Point(6,8),	cv::Scalar(92,60,24));
		koopas	= findObject(image,enemyImage2v2,	cv::Point(4,0),	cv::Scalar(136,128,0));
		floors	= findObject(image,floorImage1v2,	cv::Point(0,0),	cv::Scalar(240,252,156),cv::Rect(0,400,496,432));
		walls	= findObject(image,wallImage1v2,	cv::Point(0,0),	cv::Scalar(136,128,0),	cv::Rect(0,0,496,400));
		pipe	= findObject(image,pipeImage,		cv::Point(0,0),	cv::Scalar(16,208,128),	cv::Rect(0,0,496,400));
	}

	//Player not found
	if(playerCoords == cv::Point(-1,-1))
	{
		analyzeResult.additionalInfo = AnalyzeResult::notFound;
		return analyzeResult;
	}

	cv::Mat analyzedImage = cv::Mat(896, 512, CV_8UC3);
	//clear image
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
	int interval = 32;
	cv::Point transl = cv::Point(256-playerCoords.x,448-playerCoords.y); // translation
	cv::Point corr = cv::Point(0,0);	//correction
	cv::Point blockSize = cv::Point(interval,interval);
	rectangle(analyzedImage, cv::Rect(playerCoords+transl+cv::Point(0,6),playerCoords+cv::Point(28,32)+transl+cv::Point(0,6)),cv::Scalar(255,128,0), -1);

	//draw holes
	if(floors.size()>0)	rectangle(analyzedImage, cv::Rect(cv::Point(transl.x+interval, floors[0].y+transl.y),	cv::Point(895,floors[0].y+interval+transl.y)),	cv::Scalar(0,0,220), -1);
	corr = cv::Point(0,0);
	for(cv::Point p : goombas)	rectangle(analyzedImage, cv::Rect(p+transl+corr,p+blockSize+transl+corr),cv::Scalar(0  ,0  ,220), -1);
	for(cv::Point p : koopas)	rectangle(analyzedImage, cv::Rect(p+transl+corr,p+blockSize+transl+corr),cv::Scalar(0  ,0  ,220), -1);
	corr = cv::Point(0,0);
	for(cv::Point p : floors)	rectangle(analyzedImage, cv::Rect(p+transl,	    p+blockSize+transl	   ),cv::Scalar(100,100,100), -1);
	corr = cv::Point(2,0);
	for(cv::Point p : walls)	rectangle(analyzedImage, cv::Rect(p+transl,	    p+blockSize+transl+corr),cv::Scalar(100,100,100), -1);
	for(cv::Point p : blocks1)	rectangle(analyzedImage, cv::Rect(p+transl,	    p+blockSize+transl	   ),cv::Scalar(100,100,100), -1);
	corr = cv::Point(0,2);
	for(cv::Point p : blocks2)	rectangle(analyzedImage, cv::Rect(p+transl+corr,p+blockSize+transl+corr),cv::Scalar(100,100,100), -1);
	corr = cv::Point(-4,0);
	for(cv::Point p : blocks3)	rectangle(analyzedImage, cv::Rect(p+transl+corr,p+blockSize+transl+corr),	 cv::Scalar(100,100,100), -1);
	for(cv::Point p : pipe)		rectangle(analyzedImage, cv::Rect(p+transl,		cv::Point(p.x+64	,416)+transl),cv::Scalar(100,100,100), -1);

	//Writing data output
	std::vector<int> analyzeData;
	for(int x=interval/2; x<512; x+=interval)
	{
		for(int y=interval/3; y<896; y+=interval)
		{
			uchar* ptr = analyzedImage.ptr(y)+x*3;
			if(ptr[0]==100 && ptr[1]==100 && ptr[2]==100) analyzeData.push_back(1);
			else analyzeData.push_back(0);
		}
	}
	for(int x=interval/2; x<512; x+=interval)
	{
		for(int y=interval/3; y<896; y+=interval)
		{
			uchar* ptr = analyzedImage.ptr(y)+x*3;
			if(ptr[0]==0 && ptr[1]==0 && ptr[2]==220) analyzeData.push_back(1);
			else analyzeData.push_back(0);
			#ifdef PRINT_ANALYZED_IMAGE
				ptr[0]=255;ptr[1]=255; ptr[2]=255;
			#endif

		}
	}

	//Determine reward
	double reward = 0;
	//Checking if player is advancing according to floor position
	if(floors.size() > 0)
	{
		if(fabs(floors[0].x - oldPositionReferenceFloor)>4) reward = 50;

		if(floors[0].x != oldPositionReferenceFloor) playerVelocity.x = 1;
		else if (fabs(oldPlayerPosition - playerCoords.x) > 10) playerVelocity.x = oldPlayerPosition < playerCoords.x ? 1 : 0;
		oldPlayerPosition = playerCoords.x;
		oldPositionReferenceFloor = floors[0].x;
	}

		//Checking if player is advancing according to clouds position
	std::vector<cv::Point> clouds = findObject(image,cloudImage,	cv::Point(15,2),cv::Scalar(252,252,252),cv::Rect(0,0,496,400));
	if(clouds.size() > 0)
	{
		if(oldPositionReferenceCloud - clouds[0].x > 0)
		{
			reward = 50;
		}
		oldPositionReferenceCloud = clouds[0].x;
	}
	else oldPositionReferenceCloud = -999;

	if(fabs(playerCoords.y - oldPlayerHeight) > 4) playerVelocity.y = playerCoords.y > oldPlayerHeight ? 1 : 0;
	oldPlayerHeight = playerCoords.y;

	//obvious penalty
	AnalyzeResult::AdditionalInfo additionalInfo;
	if(!findObject(image,deadImage,cv::Point(10,4),cv::Scalar(0,148,0),cv::Rect(0,0,496,400)).empty())
	{
		reward = -100;
		additionalInfo = AnalyzeResult::killedByEnemy;
	}
//	std::cout << playerCoords.y << "  " << playerCoords.x << "\n";
	if(playerCoords.y > 375)
	{
		reward = -100;
		additionalInfo = AnalyzeResult::fallenInPitfall;
	}

	if(playerCoords.x < 20)
	{
		reward = -100;
		additionalInfo = AnalyzeResult::fallenInPitfall;
	}

	#ifdef PRINT_ANALYZED_IMAGE
		//Print
		imshow("Objects", analyzedImage);
		cv::waitKey(80);
	#endif

	analyzeResult.data = analyzeData;
	analyzeResult.reward = reward;
	analyzeResult.playerCoords = cv::Point(0,0);
	analyzeResult.playerVelocity = playerVelocity;
	analyzeResult.additionalInfo = additionalInfo;

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
std::vector<cv::Point> ImageAnalyzer::findObject(cv::Mat &image, Histogram &pattern, cv::Point offset, cv::Point size, cv::Scalar sample, cv::Scalar range)
{
	std::vector<cv::Point> result = std::vector<cv::Point>();
	int channels = image.channels();
	int width = image.cols;
	int height = image.rows;
	for(int y=size.y/2 ; y<height-size.y/2 ; y++)
	{
		for(int x=size.x/2 ; x<width-size.x/2 ; x++)
		{
			uchar* ptr = image.ptr(y)+x*channels;
			if(ptr[0]==sample[0] && ptr[1]==sample[1] && ptr[2]==sample[2])
			{
				cv::Mat segment = copyMat(image,cv::Point(x-offset.x,y-offset.y),size);
				Histogram histogram = determineHistogram(segment);
				double r1 = cv::compareHist( histogram[0], pattern[0], 3 );//CV_COMP_INTERSECT
				double r2 = cv::compareHist( histogram[1], pattern[1], 3 );//CV_COMP_INTERSECT
				double r3 = cv::compareHist( histogram[2], pattern[2], 3 );//CV_COMP_INTERSECT
				if(r1<=range[0] && r2<=range[1] && r3<=range[2])
				{
					result.push_back(cv::Point(x,y));
					rectangle(image, cv::Rect(cv::Point(x-offset.x,y-offset.y),cv::Point(x-offset.x+size.x,y-offset.y+size.y)),cv::Scalar(0,128,128), -1);
				}
			}
		}
	}
	return result;
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

/*
 *
 */
void ImageAnalyzer::printAnalyzeData(std::vector<int> sceneData, bool containsAdditionalData)
{
	int blockSize = 15;

	int imageSize = (sceneData.size())/2;
	if(containsAdditionalData) imageSize -= 5;
	int xScreenSize = sqrt(imageSize*8/14);
	int yScreenSize = xScreenSize*14/8;

	if(xScreenSize <= 0 && yScreenSize <=0) return;

	cv::Mat map = cv::Mat(blockSize*(yScreenSize+1), blockSize*(xScreenSize), CV_8UC3);

	for(int x=0; x<xScreenSize; x++)
	{
		for(int y=0; y<yScreenSize; y++)
		{
			int fieldValue = sceneData[(x)*yScreenSize+y];
			int enemyValue = sceneData[(x)*yScreenSize+y + imageSize];
			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptr = map.ptr(y*blockSize+yy)+(x*blockSize+xx)*3;
					if(fieldValue == 1) {ptr[0] = 100;ptr[1] = 100;ptr[2] = 100;}
					else if(enemyValue == 1) {ptr[0] = 220;ptr[1] = 0;ptr[2] = 0;}
					else {ptr[0] = 0;ptr[1] = 0;ptr[2] = 0;}
				}
			}
		}
	}
	if(containsAdditionalData)
	{
		for(int x=0; x<10; x++)
		{
			int infoValue = sceneData[x+imageSize+imageSize];
			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptr = map.ptr(yScreenSize*blockSize+yy)+(x*blockSize+xx)*3;
					if(infoValue == 1) {ptr[0] = 255;ptr[1] = 255;ptr[2] = 255;}
					else {ptr[0] = 0;ptr[1] = 0;ptr[2] = 0;}
				}
			}
		}
	}

	//Print
	imshow("AnalyzedSceneData", map);
	cv::waitKey(10);

}
