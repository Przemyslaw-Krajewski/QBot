/*
 * Analizator.h
 *
 *  Created on: 3 gru 2017
 *      Author: przemo
 */

#ifndef SRC_ANALYZERS_IMAGEANALYZER_H_
#define SRC_ANALYZERS_IMAGEANALYZER_H_

#include <opencv2/opencv.hpp>
#include <map>

#include "DesktopHandler.h"

//#define PRINT_ANALYZED_IMAGE

class ImageAnalyzer {

public:
	struct AnalyzeResult
	{
		cv::Mat processedImage;
		cv::Mat processedImagePast;
		cv::Mat processedImagePast2;
		bool playerFound;
		bool playerIsDead;
		bool killedByEnemy;
		bool playerWon;
	};

	typedef std::vector<cv::Mat> Histogram;

public:
	ImageAnalyzer();

	void processImage(cv::Mat* colorImage, ImageAnalyzer::AnalyzeResult *result);

	static std::vector<int> toVector(cv::Mat* image);

private:
	void reduceColors(int mask, cv::Mat* colorImage);
	void reduceColorsAndBrightness(int reduceLevel, cv::Mat* colorImage);
	void getMostFrequentInBlock(int blockSize, cv::Mat& srcImage, cv::Mat& dstImage);
	void getLeastFrequentInImage(int blockSize, cv::Mat& srcIimage, cv::Mat& dstImage);
	cv::Mat getFirst(int blockSize, cv::Mat* image);

	void calculateSituationSMB(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult);
	void calculateSituationBT(cv::Mat *image, ImageAnalyzer::AnalyzeResult *analyzeResult);

	void viewImage(int blockSize, std::string name, cv::Mat &image);
	cv::Mat cutFragment(cv::Mat* image, cv::Point leftUp, cv::Point rightDown);

	cv::Point findPlayer(cv::Mat &image);
	std::vector<cv::Point> findObject(cv::Mat &image, cv::Mat &pattern, cv::Point offset, cv::Scalar sample, cv::Rect searchBounds = cv::Rect(-1,-1,-1,-1));
	Histogram determineHistogram(cv::Mat &image);
	cv::Mat copyMat(cv::Mat src, cv::Point offset, cv::Point size);
	bool compareMat(cv::Mat &mat1, cv::Mat &mat2);
	bool compareMat(cv::Mat &image, cv::Mat &pattern, cv::Point offset);

	void markObjectInImage(cv::Mat& resultImage, cv::Point blockSize, cv::Point point, cv::Point translation, cv::Point correction, int objectType);

private:
	cv::Point imageSize;

	std::list<cv::Mat> oldImages;
	std::list<cv::Mat> oldImages2;

	Histogram playerHistogram;

	//SMB
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

	//BT
	cv::Mat emptyHealth;
	cv::Mat hair;

	cv::Point playerSize;
};

#endif /* SRC_ANALYZERS_IMAGEANALYZER_H_ */
