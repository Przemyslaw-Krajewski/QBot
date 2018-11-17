/*
 * Control.h
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#ifndef SRC_GAME_CONTROL_H_
#define SRC_GAME_CONTROL_H_

namespace Directories
{
	enum Directory
	{
		up,
		down,
		left,
		right
	};
}
using Directory = Directories::Directory;

struct Control
{
	Control(Directory t_directory)
	{
		up 		= t_directory == Directories::up;
		down 	= t_directory == Directories::down;
		left 	= t_directory == Directories::left;
		right 	= t_directory == Directories::right;
	}

	Control(int t_i)
	{
		up 		= t_i == 0;
		down 	= t_i == 1;
		right 	= t_i == 2;
		left 	= t_i == 3;
	}

	int getInt()
	{
		if(up) return 0;
		if(down) return 1;
		if(right) return 2;
		if(left) return 3;
	}

	bool up,down,left,right;
};

#endif /* SRC_GAME_CONTROL_H_ */
