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
#include "ImageAnalyzer/RawImageAnalyzer.h"
#include "ImageAnalyzer/MetaDataAnalyzer.h"
#include "MemoryAnalyzer.h"

class StateAnalyzer {

public:
	struct AnalyzeResult
	{
		cv::Mat processedImage;
		cv::Mat processedImagePast;
		cv::Mat processedImagePast2;
		double reward;
		Point playerCoords;
		Point playerVelocity;
		ScenarioAdditionalInfo additionalInfo{ScenarioAdditionalInfo::noInfo};
		bool endScenario;
	};

public:
	StateAnalyzer();
	virtual ~StateAnalyzer();

	AnalyzeResult analyze();
	State createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
												  ControllerInput& controllerInput, Point& position, Point& velocity);

	void correctScenarioHistory(std::list<SARS> &t_history, ScenarioAdditionalInfo t_additionalInfo)
		{imageAnalyzer->correctScenarioHistory(t_history, t_additionalInfo);}
	State reduceSceneState(const State& t_state, double action)
		{return imageAnalyzer->reduceSceneState(t_state, action);}

private:

	AnalyzeResult analyzeSMB();
	AnalyzeResult analyzeBT();

	ImageAnalyzer *imageAnalyzer;
	Game game;

public:
	static constexpr double WIN_REWARD     = 0.85;
	static constexpr double ADVANCE_REWARD = 0.07;
	static constexpr double LITTLE_ADVANCE_REWARD = 0.05;
	static constexpr double NOTHING_REWARD = 0.04;
	static constexpr double DIE_REWARD 	= 0.0001;
};

#endif /* SRC_ANALYZERS_STATEANALYZER_H_ */
