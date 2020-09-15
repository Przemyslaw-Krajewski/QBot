/*
 * ImageAnalyzer.h
 *
 *  Created on: 5 wrz 2020
 *      Author: przemo
 */

#ifndef SRC_ANALYZERS_IMAGEANALYZER_IMAGEANALYZER_H_
#define SRC_ANALYZERS_IMAGEANALYZER_IMAGEANALYZER_H_

#define PRINT_ANALYZED_IMAGE

#include <opencv2/opencv.hpp>
#include <map>

//TODO move somewhere else
enum class Game {BattleToads, SuperMarioBros};

class ImageAnalyzer
{
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

	ImageAnalyzer(Game t_game);
	virtual ~ImageAnalyzer();

	virtual void processImage(cv::Mat* colorImage, ImageAnalyzer::AnalyzeResult *result) = 0;

protected:

	void viewImage(int blockSize, std::string name, cv::Mat &image);
	std::vector<cv::Point> findObject(cv::Mat &image, cv::Mat &pattern, cv::Point offset, cv::Scalar sample,
			                          cv::Rect searchBounds=cv::Rect(-1,-1,-1,-1));
	bool compareMat(cv::Mat &image, cv::Mat &pattern, cv::Point offset);
	cv::Mat cutFragment(cv::Mat* image, cv::Point leftUp, cv::Point rightDown);
	std::vector<int> toVector(cv::Mat *image);

protected:

	cv::Point imageSize;
	Game game;
};



#endif /* SRC_ANALYZERS_IMAGEANALYZER_IMAGEANALYZER_H_ */
