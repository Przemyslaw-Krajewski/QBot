/*
 * Common.h
 *
 *  Created on: 28 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_BOT_COMMON_H_
#define SRC_BOT_COMMON_H_

#include <opencv2/opencv.hpp>

const int numberOfActions = 5;
const int numberOfControllerInputs = 6;

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

#endif /* SRC_BOT_COMMON_H_ */
