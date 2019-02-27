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

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>

class MemoryAnalyzer {
public:
	struct AnalyzeResult
	{
		int playerVelocityX;
		int playerVelocityY;
		int playerPositionX;
	};

public:
	MemoryAnalyzer();
	virtual ~MemoryAnalyzer();

	AnalyzeResult fetchData();

	char getMemValue(long addr);
	char setMemValue(long addr, char value);

private:
	int pid;
	off_t MEM_ADDR{0x0};
	const off_t RAM_OFFSET{0x282630};
	const off_t RAM_VEL_X_OFFSET{0x57};
	const off_t RAM_VEL_Y_OFFSET{0x9f};
	const off_t RAM_POS_X_OFFSET{0x4ac};
};

#endif /* SRC_ANALYZERS_MEMORYANALYZER_H_ */
