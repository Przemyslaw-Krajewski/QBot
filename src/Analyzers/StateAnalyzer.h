/*
 * StateAnalyzer.h
 *
 *  Created on: 7 lut 2019
 *      Author: mistrz
 */

#ifndef SRC_ANALYZERS_STATEANALYZER_H_
#define SRC_ANALYZERS_STATEANALYZER_H_

#include <vector>
#include <opencv2/opencv.hpp>

#include "../Bot/Common.h"
#include "ImageAnalyzer.h"
#include "MemoryAnalyzer.h"

class StateAnalyzer {
public:
	struct AnalyzeResult
	{
		enum AdditionalInfo {noInfo, killedByEnemy, fallenInPitfall, notFound, timeOut, won};

		cv::Mat processedImage;
		cv::Mat processedImagePast;
		double reward;
		Point playerCoords;
		Point playerVelocity;
		AnalyzeResult::AdditionalInfo additionalInfo{AdditionalInfo::noInfo};
		bool endScenario;
	};

public:
	StateAnalyzer();
	virtual ~StateAnalyzer();

	AnalyzeResult analyze();

private:
	ImageAnalyzer imageAnalyzer;

public:
	static const int WIN_REWARD     = 1000;
	static const int ADVANCE_REWARD = 50;
	static const int LITTLE_ADVANCE_REWARD = 10;
	static const int DIE_REWARD 	= -1000;
};

#endif /* SRC_ANALYZERS_STATEANALYZER_H_ */
