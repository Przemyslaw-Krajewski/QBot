/*
 * Game.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "Game.h"

#define RELEASE_CREATURE

Game::Game(int t_sizeX, int t_sizeY)
{
	robboImg = cv::imread("Grafika/Robbo.bmp", CV_LOAD_IMAGE_COLOR);
	shipImg = cv::imread("Grafika/Ship.bmp", CV_LOAD_IMAGE_COLOR);
	creatureImg = cv::imread("Grafika/Creature.bmp", CV_LOAD_IMAGE_COLOR);

	level.clear();
	levelSize = Point(t_sizeX, t_sizeY);
	std::vector<bool> row(t_sizeX,false);

	for(int i=0; i<t_sizeY; i++)
	{
		level.push_back(row);
	}
	reset();
}

Game::~Game()
{

}

void Game::reset()
{
	robboCoords = Point(0, 0);
	shipCoords = Point(levelSize.x-1, levelSize.y-1);
	creatureCoords = Point(levelSize.x-1, levelSize.y-2);
	time = 200;

}

std::pair<double,bool> Game::execute(Control t_control)
{
	bool bumped = false;

//	time--;
//	if(time < 1) return std::pair<double,bool>(0.004,true);

	if(t_control.up)
	{
		if(robboCoords.y > 0) robboCoords.y--;
		else bumped = true;
	}
	if(t_control.down)
	{
		if(robboCoords.y < levelSize.y-1) robboCoords.y++;
		else bumped = true;
	}
	if(t_control.left)
	{
		if(robboCoords.x > 0) robboCoords.x--;
		else bumped = true;
	}
	if(t_control.right)
	{
		if(robboCoords.x < levelSize.x-1) robboCoords.x++;
		else bumped = true;
	}

	if(level[robboCoords.x][robboCoords.y])
	{
		if(t_control.up) robboCoords.y++;
		if(t_control.down) robboCoords.y--;
		if(t_control.left) robboCoords.x++;
		if(t_control.right)robboCoords.x--;
		bumped = true;
	}

#ifdef RELEASE_CREATURE
	for(int i=0; i<1 ;i++)
	{
		if(rand()%2 && creatureCoords.x < robboCoords.x && creatureCoords.x < levelSize.x && !level[creatureCoords.x+1][creatureCoords.y])
			creatureCoords.x++;
		else if(creatureCoords.x > robboCoords.x && creatureCoords.x > 0		   && !level[creatureCoords.x-1][creatureCoords.y])
			creatureCoords.x--;
		else if(creatureCoords.y < robboCoords.y && creatureCoords.y < levelSize.y && !level[creatureCoords.x][creatureCoords.y+1])
			creatureCoords.y++;
		else if(creatureCoords.y > robboCoords.y && creatureCoords.y > 0 		   && !level[creatureCoords.x][creatureCoords.y-1])
			creatureCoords.y--;
		else if(creatureCoords.x < robboCoords.x && creatureCoords.x < levelSize.x && !level[creatureCoords.x+1][creatureCoords.y])
			creatureCoords.x++;
	}
#endif

	//Reached ship
	if(shipCoords == robboCoords) return std::pair<double,bool>(100,true);
	//Eaten by creature
#ifdef RELEASE_CREATURE
		if(fabs(creatureCoords.x-robboCoords.x)<2 && fabs(creatureCoords.y-robboCoords.y)<2) return std::pair<double,bool>(-100,true);
#endif
	//Hit wall
	if(bumped) return std::pair<double,bool>(-10,false);
	//Nothing
	return std::pair<double,bool>(0,false);
}

std::vector<int> Game::getState()
{
	std::vector<int> result;
	result.push_back(robboCoords.x);
	result.push_back(robboCoords.y);
#ifdef RELEASE_CREATURE
	result.push_back(creatureCoords.x);
	result.push_back(creatureCoords.y);
#endif
	return result;
}

void Game::display()
{
	cv::Mat img;
	int fieldSize = 50;
	img = cv::Mat(levelSize.x*fieldSize+1, levelSize.y*fieldSize+1, CV_8UC3);


	for(int y = 0 ; y < img.rows ; y++)
	{
		uchar* ptr = img.ptr((int)y);
		for(int x = 0 ; x < img.cols*3 ; x++)
		{
			//Not optimal
			if(y%fieldSize && (x/3)%fieldSize) *ptr=0;
			else *ptr=255;

			int blockX = x/(fieldSize)/3;
			int blockY = y/fieldSize;

			if(blockX < levelSize.x && blockY < levelSize.y && level[blockX][blockY]) *ptr = 255;

			ptr = ptr+1;
		}
	}

	for(int x = 0 ; x < robboImg.cols ; x++)
	{
		for(int y = 0 ; y < robboImg.rows ; y++)
		{
			uchar* ptr1 = robboImg.ptr((int)y)+((int)x)*3;
			uchar* ptr2 = img.ptr((int)y+(int)robboCoords.y*fieldSize)+((int)x+(int)robboCoords.x*fieldSize)*3;
			if(ptr1[0]==0 && ptr1[1]==0 && ptr1[2]==0) continue;
			ptr2[0] = ptr1[0];
			ptr2[1] = ptr1[1];
			ptr2[2] = ptr1[2];
		}
	}

	for(int x = 0 ; x < shipImg.cols ; x++)
	{
		for(int y = 0 ; y < shipImg.rows ; y++)
		{
			uchar* ptr1 = shipImg.ptr((int)y)+((int)x)*3;
			uchar* ptr2 = img.ptr((int)y+(int)shipCoords.y*fieldSize+1)+((int)x+(int)shipCoords.x*fieldSize+1)*3;
			if(ptr1[0]==0 && ptr1[1]==0 && ptr1[2]==0) continue;
			ptr2[0] = ptr1[0];
			ptr2[1] = ptr1[1];
			ptr2[2] = ptr1[2];
		}
	}

#ifdef RELEASE_CREATURE
	for(int x = 0 ; x < creatureImg.cols ; x++)
	{
		for(int y = 0 ; y < creatureImg.rows ; y++)
		{
			uchar* ptr1 = creatureImg.ptr((int)y)+((int)x)*3;
			uchar* ptr2 = img.ptr((int)y+(int)creatureCoords.y*fieldSize+1)+((int)x+(int)creatureCoords.x*fieldSize+1)*3;
			if(ptr1[0]==0 && ptr1[1]==0 && ptr1[2]==0) continue;
			ptr2[0] = ptr1[0];
			ptr2[1] = ptr1[1];
			ptr2[2] = ptr1[2];
		}
	}
#endif

	imshow("Robbo", img);
	cv::waitKey(20);
}
