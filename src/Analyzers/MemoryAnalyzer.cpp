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

	if(pidFile.is_open() && !pidFile.eof())	pidFile >> pid;
	else assert("pid.txt management error" && 0);

	pidFile.close();
	system("rm pid.txt");

	if(pid == -1) assert("FCEU pid not found" && 0);

	char memCommand[64];
	sprintf(memCommand, "cat /proc/%ld/maps | grep heap | cut -c -12 > mem.txt", (long)pid);
	system(memCommand);
	std::ifstream memFile;
	memFile.open ("mem.txt");

	if(memFile.is_open() && !memFile.eof()) memFile >> std::hex >> MEM_ADDR;
	else assert("mem.txt management error" && 0);

	memFile.close();
	system("rm mem.txt");

	if(MEM_ADDR == 0x0) assert("FCEU mem not found" && 0);

	std::cout << pid << " " << MEM_ADDR << "\n";
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

	result.playerVelocityX = (int) getMemValue(MEM_ADDR+RAM_OFFSET+RAM_VEL_X_OFFSET);
	result.playerVelocityY = (int) getMemValue(MEM_ADDR+RAM_OFFSET+RAM_VEL_Y_OFFSET);
	result.playerPositionX = (int) getMemValue(MEM_ADDR+RAM_OFFSET+RAM_POS_X_OFFSET);

	return result;
}
