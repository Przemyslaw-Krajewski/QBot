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

#include "ImageAnalyzer.h"
#include "MemoryAnalyzer.h"

class StateAnalyzer {
public:

	struct Point
	{
		Point() {x=y=0;}
		Point(int t_x, int t_y) {x=t_x;y=t_y;}
		Point& operator=(const cv::Point& t_p ) {x=t_p.x;y=t_p.y;return *this;}
		Point& operator=(const Point & t_p ) {x=t_p.x;y=t_p.y;return *this;}
		bool operator==(const Point & t_p ) {return (x==t_p.x && y==t_p.y);}
		int x;
		int y;
	};

	struct AnalyzeResult
	{
		enum AdditionalInfo {noInfo, killedByEnemy, fallenInPitfall, notFound, timeOut};

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

	static void printAnalyzeData(AnalyzeResult& sceneData);

private:
	ImageAnalyzer imageAnalyzer;
	MemoryAnalyzer memoryAnalyzer;
};

#endif /* SRC_ANALYZERS_STATEANALYZER_H_ */
