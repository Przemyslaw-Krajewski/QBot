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
#include <assert.h>

//#define PRINT_FETCHED_SCREEN

class MemoryAnalyzer {
public:
	struct AnalyzeResult
	{
		int playerVelocityX;
		int playerVelocityY;
		int playerPositionX;
	};

private:
	MemoryAnalyzer();
	~MemoryAnalyzer();

public:
	static MemoryAnalyzer* getPtr();

	AnalyzeResult fetchData();

	char getMemValue(long addr);
	char setMemValue(long addr, char value);

	cv::Mat fetchScreenData();

private:
	static MemoryAnalyzer* ptr;

	int pid;
	off_t MEM_ADDR;
	const off_t RAM_OFFSET{0x282630};
	const off_t RAM_VEL_X_OFFSET{0x57};
	const off_t RAM_VEL_Y_OFFSET{0x9f};
	const off_t RAM_POS_X_OFFSET{0x4ac};
	const off_t RAM_ADDR{0x555555b8c610};
	const off_t XBUFF_ADDR{0x555555b5c5b0};
	const off_t PALETTE_ADDR{0x555555ac59c0};
};

#endif /* SRC_ANALYZERS_MEMORYANALYZER_H_ */
