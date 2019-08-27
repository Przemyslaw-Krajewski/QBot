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

		cv::Mat fieldAndEnemiesLayout;
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
	static constexpr double WIN_REWARD     = 0.4;
	static constexpr double ADVANCE_REWARD = 0.15;
	static constexpr double LITTLE_ADVANCE_REWARD = 0.12;
	static constexpr double DIE_REWARD 	= 0.0001;
};

#endif /* SRC_ANALYZERS_STATEANALYZER_H_ */
