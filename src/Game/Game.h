/*
 * Game.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_GAME_GAME_H_
#define SRC_GAME_GAME_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

#include "Point.h"
#include "Control.h"

class Game {
public:
	Game(int t_sizeX, int t_sizeY);
	virtual ~Game();

	void reset();

	std::pair<double,bool> execute(Control t_control);
	void display();

	std::vector<int> getState();
	Point getCreatureCoords() {return creatureCoords;}

	void setLevel(std::vector<std::vector<bool>> t_level) {level = t_level;}

private:
	Point levelSize;
	std::vector<std::vector<bool>> level;
	Point robboCoords;
	Point shipCoords;
	Point creatureCoords;
	int time;

	cv::Mat robboImg;
	cv::Mat shipImg;
	cv::Mat creatureImg;
};

#endif /* SRC_GAME_GAME_H_ */
