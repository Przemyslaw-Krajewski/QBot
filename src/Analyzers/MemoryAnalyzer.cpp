/*
 * MemoryAnalyzer.cpp
 *
 *  Created on: 17 lut 2019
 *      Author: mistrz
 */

#include "MemoryAnalyzer.h"

/*
 *
 */
MemoryAnalyzer::MemoryAnalyzer() {

	pid = -1;
	system("ps -a | grep fceux | sed -e 's, ,,g' | sed -e 's,[a-zA-Z].*,,g' > pid.txt");
	std::ifstream pidFile;
	pidFile.open ("pid.txt");

	if(pidFile.is_open() && !pidFile.eof())
	{
		pidFile >> pid;
		pidFile.close();
		system("rm pid.txt");
	}
	else
	{
		assert("pid.txt management error" && 0);
		system("rm pid.txt");
	}

	if(pid == -1)
	{
		assert("FCEU pid not found" && 0);
	}

	std::cout << pid << "\n";
}

/*
 *
 */
MemoryAnalyzer::~MemoryAnalyzer() {

}

/*
 *
 */
char MemoryAnalyzer::getMemValue(long addr)
{

	char file[64];
	sprintf(file, "/proc/%ld/mem", (long)pid);
	int fd = open(file, O_RDWR);
	if(fd == -1) throw std::string("Could not read memory data");

	ptrace(PTRACE_ATTACH, pid, 0, 0);
	waitpid(pid, NULL, 0);

	char value = 10;
	pread(fd, &value, sizeof(value), addr);

	ptrace(PTRACE_DETACH, pid, 0, 0);
	close(fd);

	return value;

}

/*
 *
 */
char MemoryAnalyzer::setMemValue(long addr, char value)
{
	char file[64];
	sprintf(file, "/proc/%ld/mem", (long)pid);
	int fd = open(file, O_RDWR);

	ptrace(PTRACE_ATTACH, pid, 0, 0);
	waitpid(pid, NULL, 0);

	pwrite(fd, &value, sizeof(value), addr);

	ptrace(PTRACE_DETACH, pid, 0, 0);
	close(fd);

	return value;

}

/*
 *
 */
MemoryAnalyzer::AnalyzeResult MemoryAnalyzer::fetchData()
{
	AnalyzeResult result;

	result.playerVelocityX = (int) getMemValue(RAM_ADDR+RAM_VEL_X_ADDR);
	result.playerVelocityY = (int) getMemValue(RAM_ADDR+RAM_VEL_Y_ADDR);
	result.playerPositionX = (int) getMemValue(RAM_ADDR+RAM_POS_X_ADDR);

	return result;
}
