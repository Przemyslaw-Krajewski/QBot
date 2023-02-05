/*
 * ImageAnalyzer.h
 *
 *  Created on: 5 wrz 2020
 *      Author: przemo
 */

#ifndef SRC_ANALYZERS_IMAGEANALYZER_IMAGEANALYZER_H_
#define SRC_ANALYZERS_IMAGEANALYZER_IMAGEANALYZER_H_

//#define PRINT_ANALYZED_IMAGE
//#define PRINT_REDUCED_IMAGE

#include <opencv2/opencv.hpp>
#include <map>
#include <functional>

#include "Point.h"
#include "../../Bot/State.h"
#include "../../Bot/Controller.h"

#include "../../Loggers/DataDrawer.h"


enum class Game {BattleToads, SuperMarioBros};
using ReduceStateMethod = std::function<State(State&)>;

class ImageAnalyzer
{
public:

	struct AnalyzeResult
	{
		std::vector<cv::Mat> processedImages;
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
	virtual State createSceneState(std::vector<cv::Mat> &t_images, ControllerInput& t_controllerInput, Point& t_position, Point& t_velocity) = 0;
	virtual void correctScenarioHistory(std::list<SARS> &t_history, bool t_killedByEnemy);

	ReduceStateMethod getReduceStateMethod() {return reducedStateMethod;}
protected:

	void viewImage(int blockSize, std::string name, cv::Mat &image);
	std::vector<cv::Point> findObject(cv::Mat &image, cv::Mat &pattern, cv::Point offset, cv::Scalar sample,
			                          cv::Rect searchBounds=cv::Rect(-1,-1,-1,-1));
	bool compareMat(cv::Mat &image, cv::Mat &pattern, cv::Point offset);
	cv::Mat cutFragment(cv::Mat* image, cv::Point leftUp, cv::Point rightDown);

protected:

	cv::Point imageSize;
	Game game;

	ReduceStateMethod reducedStateMethod = nullptr;

	const int MAX_INPUT_VALUE = 1;
	const int MIN_INPUT_VALUE = 0;
};


#endif /* SRC_ANALYZERS_IMAGEANALYZER_IMAGEANALYZER_H_ */
