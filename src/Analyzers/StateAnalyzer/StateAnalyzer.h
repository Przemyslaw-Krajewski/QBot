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

enum class ScenarioAdditionalInfo {ok, killedByEnemy, killedByEnvironment, playerNotFound, timeOut, won};

class StateAnalyzer
{

public:

	struct AnalyzeResult
	{
		double reward;
		Point playerCoords;
		Point playerVelocity;
		ScenarioAdditionalInfo scenarioStatus = ScenarioAdditionalInfo::ok;
		State processedState;

		bool isPlayerFound() {return scenarioStatus != ScenarioAdditionalInfo::playerNotFound;}
		bool endScenario() {return scenarioStatus != ScenarioAdditionalInfo::ok;}
	};

public:
	StateAnalyzer();
	virtual ~StateAnalyzer();

	AnalyzeResult analyze(ControllerInput &t_input);

	void resetTimeLimit() {timeLimit = TIME_LIMIT;}

	ReduceStateMethod getReduceStateMethod() {return (imageAnalyzer->getReduceStateMethod());}

	void correctScenarioHistory(std::list<SARS> &t_history, ScenarioAdditionalInfo t_additionalInfo);

private:

	AnalyzeResult analyzeSMB(ControllerInput &t_input);
	AnalyzeResult analyzeBT(ControllerInput &t_input);
	bool handleTimeLimit(double t_gainedReward);

	ImageAnalyzer *imageAnalyzer;
	Game game;

	int timeLimit;

	int screenPosition = 0;
	int score = -1;

public:
	static constexpr double CHECKPOINT_REWARD     = 0.35;
	static constexpr double GREAT_ADVANCE_REWARD  = 0.880;
	static constexpr double ADVANCE_REWARD 		  = 0.875;
	static constexpr double LITTLE_ADVANCE_REWARD = 0.75;
	static constexpr double NOTHING_REWARD 		  = 0.625;

	static constexpr double WIN_REWARD     		  = 0.90;
	static constexpr double DIE_REWARD 			  = 0.00001;
	static constexpr double TIMEOUT_REWARD 		  = 0.2;

	static constexpr int TIME_LIMIT = 40;
};

#endif /* SRC_ANALYZERS_STATEANALYZER_H_ */
