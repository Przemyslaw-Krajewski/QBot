/*
 * DataPrinter.h
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_LOGGERS_DATADRAWER_H_
#define SRC_LOGGERS_DATADRAWER_H_

#include <opencv2/opencv.hpp>

#include "../Bot/Controller.h"
#include "../Bot/State.h"

class DataDrawer {
private:
	DataDrawer();

public:
	static void drawState(State t_State, StateInfo t_stateInfo, std::string name);
	static void drawReducedState(State t_reducedState);
	static void drawAdditionalInfo(double t_reward, double t_maxTime, double t_time, ControllerInput t_keys, bool pressedKey);
private:
	inline static void drawBlock(cv::Mat *mat, int t_blockSize, Point t_point, cv::Scalar t_color);
	inline static void drawBorderedBlock(cv::Mat *mat, int t_blockSize, Point t_point, cv::Scalar t_color);
	inline static void drawBar(cv::Mat *mat, int t_barHeight, int t_barWidth, double progress, cv::Point t_point, cv::Scalar t_color);
};

#endif /* SRC_LOGGERS_DATADRAWER_H_ */
