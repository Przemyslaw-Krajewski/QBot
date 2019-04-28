/*
 * DataPrinter.h
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_LOGGERS_DATADRAWER_H_
#define SRC_LOGGERS_DATADRAWER_H_

#include "../Analyzers/StateAnalyzer.h"

class DataDrawer {
private:
	DataDrawer();

public:
	static void drawAnalyzedData(StateAnalyzer::AnalyzeResult& t_sceneData, std::vector<bool> t_keys);
private:
	inline static void drawBlock(cv::Mat *mat, int t_blockSize, StateAnalyzer::Point t_point, cv::Scalar t_color);
	inline static void drawBorderedBlock(cv::Mat *mat, int t_blockSize, StateAnalyzer::Point t_point, cv::Scalar t_color);
};

#endif /* SRC_LOGGERS_DATADRAWER_H_ */
