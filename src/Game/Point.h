/*
 * Point.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_GAME_POINT_H_
#define SRC_GAME_POINT_H_

struct Point
{
	Point()
	{
		x=0;
		y=0;
	}
	Point(int t_x, int t_y)
	{
		x=t_x;
		y=t_y;
	}

	bool operator==(Point t_p)
	{
		return x==t_p.x && y==t_p.y;
	}

	int x;
	int y;
};

#endif /* SRC_GAME_POINT_H_ */
