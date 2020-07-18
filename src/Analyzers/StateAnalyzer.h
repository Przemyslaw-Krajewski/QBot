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
	constexpr static double WIN_REWARD = 0.8;
	constexpr static double ADVANCE_REWARD = 0.02;
	constexpr static double LITTLE_ADVANCE_REWARD = 0.01;
	constexpr static double NOTHING_REWARD = 0.005;
	constexpr static double DIE_REWARD = 0.00001;
	constexpr static double JUMP_HOLD_PENALTY = 0.002;
};

#endif /* SRC_ANALYZERS_STATEANALYZER_H_ */
