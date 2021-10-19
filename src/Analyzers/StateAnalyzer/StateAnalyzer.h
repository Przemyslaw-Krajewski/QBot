/*
 * StateAnalyzer.h
 *
 *  Created on: 7 lut 2019
 *      Author: mistrz
 */

#ifndef SRC_ANALYZERS_STATEANALYZER_H_
#define SRC_ANALYZERS_STATEANALYZER_H_

#include <vector>
#include <functional>

#include "../../Bot/State.h"
#include "../../Bot/Controller.h"
#include "../ImageAnalyzer/RawImageAnalyzer.h"
#include "../ImageAnalyzer/MetaDataAnalyzer.h"
#include "../MemoryAnalyzer.h"

#include "../../Loggers/DataDrawer.h"

class StateAnalyzer {

public:

	struct AnalyzeResult
	{
		double reward;
		Point playerCoords;
		Point playerVelocity;
		ScenarioAdditionalInfo scenarioStatus = ScenarioAdditionalInfo::ok;
		State processedState;
		StateInfo stateInfo;

		bool isPlayerFound() {return scenarioStatus != ScenarioAdditionalInfo::playerNotFound;}
		bool endScenario() {return scenarioStatus != ScenarioAdditionalInfo::ok;}
	};

public:
	StateAnalyzer();
	virtual ~StateAnalyzer();

	AnalyzeResult analyze(ControllerInput &t_input);

	void resetTimeLimit() {timeLimit = TIME_LIMIT;}

	ReduceStateMethod getReduceStateMethod() {return (imageAnalyzer->getReduceStateMethod());}

	void correctScenarioHistory(std::list<SARS> &t_history, ScenarioAdditionalInfo t_additionalInfo)
		{imageAnalyzer->correctScenarioHistory(t_history, t_additionalInfo);}

private:

	AnalyzeResult analyzeSMB(ControllerInput &t_input);
	AnalyzeResult analyzeBT(ControllerInput &t_input);
	bool handleTimeLimit(double t_gainedReward);

	ImageAnalyzer *imageAnalyzer;
	Game game;

	int timeLimit;

public:
	static constexpr double WIN_REWARD     = 0.85;
	static constexpr double ADVANCE_REWARD = 0.07;
	static constexpr double LITTLE_ADVANCE_REWARD = 0.05;
	static constexpr double NOTHING_REWARD = 0.04;
	static constexpr double DIE_REWARD 	= 0.0001;

	static constexpr int TIME_LIMIT = 150;
};

#endif /* SRC_ANALYZERS_STATEANALYZER_H_ */
