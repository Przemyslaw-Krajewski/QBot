/*
 * MemoryAnalyzer.cpp
 *
 *  Created on: 17 lut 2019
 *      Author: mistrz
 */

#include "MemoryAnalyzer.h"

MemoryAnalyzer* MemoryAnalyzer::ptr = nullptr;

/*
 *
 */
MemoryAnalyzer* MemoryAnalyzer::getPtr()
{
	if(ptr == nullptr) ptr = new MemoryAnalyzer();
	return ptr;
}

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

	MEM_ADDR = 0x0;
	if(memFile.is_open() && !memFile.eof()) memFile >> std::hex >> MEM_ADDR;
	else assert("mem.txt management error" && 0);

	memFile.close();
	system("rm mem.txt");

	if(MEM_ADDR == 0x0) assert("FCEU mem not found" && 0);

	std::cout << "PID: "<< pid << " MEM ADDR:" << MEM_ADDR << "\n";
}

/*
 *
 */
MemoryAnalyzer::~MemoryAnalyzer() {

}

/*
 *
 */
unsigned char MemoryAnalyzer::getMemValue(long addr)
{

	char file[64];
	sprintf(file, "/proc/%ld/mem", (long)pid);
	int fd = open(file, O_RDWR);
	if(fd == -1) throw std::string("Could not read memory data");

	ptrace(PTRACE_ATTACH, pid, 0, 0);
	waitpid(pid, NULL, 0);

	unsigned char value = 10;
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

	result.playerVelocityX = (unsigned int) getMemValue(RAM_ADDR+RAM_VEL_X_OFFSET);
	result.playerVelocityY = (unsigned int) getMemValue(RAM_ADDR+RAM_VEL_Y_OFFSET);
	result.playerPositionX = (unsigned int) getMemValue(RAM_ADDR+RAM_POS_X_OFFSET);
	result.playerPositionY = (unsigned int) getMemValue(RAM_ADDR+RAM_POS_Y_OFFSET);

	return result;
}

/*
 *
 */
cv::Mat MemoryAnalyzer::fetchScreenData()
{
	int xScreenSize = 256;
	int yScreenSize = 256;

	cv::Mat screenResult = cv::Mat((yScreenSize+1), (xScreenSize), CV_8UC3);
	char imageData[256*256];
	char paletteData[256*4];

	char file[64];
	sprintf(file, "/proc/%ld/mem", (long)pid);
	int fd = open(file, O_RDWR);
	if(fd == -1) throw std::string("Could not read memory data");

	ptrace(PTRACE_ATTACH, pid, 0, 0);
	waitpid(pid, NULL, 0);

	pread(fd, imageData, 256*256*sizeof(char), XBUFF_ADDR);
	pread(fd, paletteData, 256*4*sizeof(char), PALETTE_ADDR);

	ptrace(PTRACE_DETACH, pid, 0, 0);
	close(fd);

	for(int y=0, ys=0; y<yScreenSize; ys+=xScreenSize, y++)
	{
		for(int x=0; x<xScreenSize; x++)
		{
			unsigned char value = imageData[ys+x];

			uchar* ptr = screenResult.ptr(y)+(x)*3;
			ptr[0] = paletteData[value*4+2]; // blue
			ptr[1] = paletteData[value*4+1]; // green
			ptr[2] = paletteData[value*4+0]; // red
		}
	}

#ifdef PRINT_FETCHED_SCREEN
	//Print
	imshow("Video from memory", screenResult);
	cv::waitKey(10);
#endif

	return screenResult;
}

/*
 *
 */
cv::Mat MemoryAnalyzer::fetchRawScreenData()
{
	int blockSize = 2;

	int xScreenSize = 256;
	int yScreenSize = 256;

	cv::Mat screenResult = cv::Mat(blockSize*(yScreenSize+1), blockSize*(xScreenSize), CV_8UC3);
	char imageData[256*256];

	char file[64];
	sprintf(file, "/proc/%ld/mem", (long)pid);
	int fd = open(file, O_RDWR);
	if(fd == -1) throw std::string("Could not read memory data");

	ptrace(PTRACE_ATTACH, pid, 0, 0);
	waitpid(pid, NULL, 0);

	pread(fd, imageData, 256*256*sizeof(char), XBUFF_ADDR);

	ptrace(PTRACE_DETACH, pid, 0, 0);
	close(fd);

	for(int x=0; x<xScreenSize; x++)
	{
		for(int y=0; y<yScreenSize; y++)
		{
			int value = imageData[y*xScreenSize+x];
			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptr = screenResult.ptr(y*blockSize+yy)+(x*blockSize+xx)*3;
					ptr[0] = value;
					ptr[1] = value;
					ptr[2] = value;
				}
			}
		}
	}

#ifdef PRINT_FETCHED_SCREEN
	//Print
	imshow("Video from memory", screenResult);
	cv::waitKey(10);
#endif

	return screenResult;
}
