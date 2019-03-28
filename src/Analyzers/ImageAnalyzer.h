/*
 * Analizator.h
 *
 *  Created on: 3 gru 2017
 *      Author: przemo
 */

#ifndef SRC_ANALYZERS_IMAGEANALYZER_H_
#define SRC_ANALYZERS_IMAGEANALYZER_H_

#include <opencv2/opencv.hpp>

#include "DesktopHandler.h"

//#define PRINT_ANALYZED_IMAGE

class ImageAnalyzer {

public:
	struct AnalyzeResult
	{
		cv::Mat fieldAndEnemiesLayout;
		bool playerFound;
		bool playerIsDead;
		bool killedByEnemy;
		bool playerWon;
	};

	typedef std::vector<cv::Mat> Histogram;

public:
	ImageAnalyzer();

	AnalyzeResult processImage(cv::Mat* image);

private:
	cv::Point findPlayer(cv::Mat &image);
	std::vector<cv::Point> findObject(cv::Mat &image, cv::Mat &pattern, cv::Point offset, cv::Scalar sample, cv::Rect searchBounds = cv::Rect(-1,-1,-1,-1));
	Histogram determineHistogram(cv::Mat &image);
	cv::Mat copyMat(cv::Mat src, cv::Point offset, cv::Point size);
	bool compareMat(cv::Mat &mat1, cv::Mat &mat2);
	bool compareMat(cv::Mat &image, cv::Mat &pattern, cv::Point offset);

	void markObjectInImage(cv::Mat& resultImage, cv::Point blockSize, cv::Point point, cv::Point translation, cv::Point correction, int objectType);

private:
	cv::Point imageSize;

	Histogram playerHistogram;

	cv::Mat enemyImage1,enemyImage1v2;
	cv::Mat enemyImage2,enemyImage2v2;
	cv::Mat floorImage1,floorImage1v2;
	cv::Mat wallimage1,wallImage1v2;
	cv::Mat blockImage1,blockImage1v2;
	cv::Mat blockImage2,blockImage2v2;
	cv::Mat blockImage3,blockImage3v2;
	cv::Mat pipeImage;
	cv::Mat deadImage;
	cv::Mat winImage;
	cv::Mat mushroomImage;
	cv::Mat cloudImage;

	cv::Point playerSize;
};

#endif /* SRC_ANALYZERS_IMAGEANALYZER_H_ */
