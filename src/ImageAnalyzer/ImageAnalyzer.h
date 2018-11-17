/*
 * Analizator.h
 *
 *  Created on: 3 gru 2017
 *      Author: przemo
 */

#ifndef SRC_IMAGEANALYZER_IMAGEANALYZER_H_
#define SRC_IMAGEANALYZER_IMAGEANALYZER_H_

#include <opencv2/opencv.hpp>

#include "../ImageAnalyzer/DesktopHandler.h"

//#define PRINT_ANALYZED_IMAGE

class ImageAnalyzer {

public:
	struct AnalyzeResult
	{
		enum AdditionalInfo {noInfo,fallenInPitfall,killedByEnemy,notFound};

		std::vector<int> data;
		double reward;
		cv::Point playerCoords;
		cv::Point playerVelocity;
		AdditionalInfo additionalInfo{noInfo};
	};

	typedef std::vector<cv::Mat> Histogram;

public:
	ImageAnalyzer();

	AnalyzeResult processImage();

	static void printAnalyzeData(std::vector<int> sceneData, bool containsAdditionalData);

private:
	cv::Point findPlayer(cv::Mat &image);
	std::vector<cv::Point> findObject(cv::Mat &image, Histogram &pattern, cv::Point offset, cv::Point size, cv::Scalar sample, cv::Scalar range);
	std::vector<cv::Point> findObject(cv::Mat &image, cv::Mat &pattern, cv::Point offset, cv::Scalar sample, cv::Rect searchBounds = cv::Rect(-1,-1,-1,-1));
	Histogram determineHistogram(cv::Mat &image);
	cv::Mat copyMat(cv::Mat src, cv::Point offset, cv::Point size);
	bool compareMat(cv::Mat &mat1, cv::Mat &mat2);
	bool compareMat(cv::Mat &image, cv::Mat &pattern, cv::Point offset);

private:
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
	cv::Mat mushroomImage;
	cv::Mat cloudImage;

	cv::Point playerSize;

	int oldPositionReferenceFloor;
	int oldPositionReferenceCloud;
	int oldPlayerPosition;
	int oldPlayerHeight;
};

#endif /* SRC_IMAGEANALYZER_IMAGEANALYZER_H_ */
