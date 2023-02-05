/*
 * MemoryAnalyzer.h
 *
 *  Created on: 17 lut 2019
 *      Author: mistrz
 */

#ifndef SRC_ANALYZERS_MEMORYANALYZER_H_
#define SRC_ANALYZERS_MEMORYANALYZER_H_

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

//#define PRINT_FETCHED_SCREEN

class MemoryAnalyzer {
public:
	struct AnalyzeResult
	{
		int playerVelocityX;
		int playerVelocityY;
		int playerPositionX;
		int playerPositionY;
		int obstaclePositionX;
		int obstaclePositionZ;
		unsigned int screenVelocity;
		unsigned int screenPosition;
		unsigned int score;
	};

private:
	MemoryAnalyzer();
	~MemoryAnalyzer();

public:
	static MemoryAnalyzer* getPtr();

	AnalyzeResult fetchData();
	AnalyzeResult fetchDataBT();

	off_t getMemValue(off_t addr, size_t size);
	char setMemValue(off_t addr, char value);

	cv::Mat fetchScreenData();
	cv::Mat fetchRawScreenData();
	void setController(int c);
	void loadState();

private:
	static MemoryAnalyzer* ptr;

	int pid;
	off_t HEAP_ADDR;

	const off_t RAM_POS_X_BT_OFFSET{0x41c};
	const off_t RAM_POS_Y_BT_OFFSET{0x43a};
	const off_t RAM_POS_X_OBST_BT_OFFSET{0x41f};
	const off_t RAM_POS_Z_OBST_BT_OFFSET{0x488};

	const off_t RAM_VEL_X_OFFSET{0x57};
	const off_t RAM_VEL_Y_OFFSET{0x9f};
	const off_t RAM_POS_X_OFFSET{0x4ac};
	const off_t RAM_POS_Y_OFFSET{0x0ce};
	const off_t RAM_POS_SCREEN_OFFSET{0x71c};
	const off_t RAM_VEL_SCREEN_OFFSET{0x775};
	const off_t RAM_SCORE_OFFSET{0x7e7};

	const off_t RAMPTR_ADDR_OFFSET{0x108578}; // RAMptr
	const off_t XBUFFPTR_ADDR_OFFSET{0x13e768}; // XBackBuffptr

	off_t RAM_ADDR;
	off_t XBUFF_ADDR;
	const off_t PALETTE_ADDR{0xbcef00}; //s_psdl
	const off_t CONTROL_ADDR{0x10c1c7}; //sterowanie
	const off_t LOADSTATE_ADDR{0x10c1c6}; //wczytaj
};

#endif /* SRC_ANALYZERS_MEMORYANALYZER_H_ */
