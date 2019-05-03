/*
 * DataPrinter.h
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_LOGGERS_DATADRAWER_H_
#define SRC_LOGGERS_DATADRAWER_H_

#include <opencv2/opencv.hpp>

#include "../Bot/Common.h"
#include "../Analyzers/StateAnalyzer.h"

class DataDrawer {
private:
	DataDrawer();

public:
	static void drawAnalyzedData(StateAnalyzer::AnalyzeResult& t_sceneData, ControllerInput t_keys);
private:
	inline static void drawBlock(cv::Mat *mat, int t_blockSize, Point t_point, cv::Scalar t_color);
	inline static void drawBorderedBlock(cv::Mat *mat, int t_blockSize, Point t_point, cv::Scalar t_color);
};

#endif /* SRC_LOGGERS_DATADRAWER_H_ */