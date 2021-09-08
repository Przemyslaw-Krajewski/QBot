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
	};

private:
	MemoryAnalyzer();
	~MemoryAnalyzer();

public:
	static MemoryAnalyzer* getPtr();

	AnalyzeResult fetchData();

	unsigned long getMemValue(long addr, size_t size);
	char setMemValue(long addr, char value);

	cv::Mat fetchScreenData();
	cv::Mat fetchRawScreenData();
	void setController(int c);
	void loadState();

private:
	static MemoryAnalyzer* ptr;

	int pid;
	off_t MEM_ADDR;

	const off_t RAM_VEL_X_OFFSET{0x57};
	const off_t RAM_VEL_Y_OFFSET{0x9f};
	const off_t RAM_POS_X_OFFSET{0x4ac};
	const off_t RAM_POS_Y_OFFSET{0x0ce};

	const off_t RAMPTR_ADDR_OFFSET{0x12f198}; // RAMptr 0x555555839198
	const off_t XBUFFPTR_ADDR_OFFSET{0x1323c8}; // XBackBuffptr 0x55555583c3c8

	off_t RAM_ADDR;
	off_t XBUFF_ADDR;
	const off_t PALETTE_ADDR{0x5555558c5a80}; //s_psdl
	const off_t CONTROL_ADDR{0x55555583c747}; //sterowanie
	const off_t LOADSTATE_ADDR{0x55555583c746}; //wczytaj
};

#endif /* SRC_ANALYZERS_MEMORYANALYZER_H_ */
