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
	else throw std::string("pid.txt management error");

	pidFile.close();
	system("rm pid.txt");

	if(pid == -1) throw std::string("FCEU pid not found");

	char memCommand[64];
	sprintf(memCommand, "cat /proc/%ld/maps | grep heap | cut -c -12 > mem.txt", (long)pid);
	system(memCommand);
	std::ifstream memFile;
	memFile.open ("mem.txt");

	HEAP_ADDR = 0x0;
	if(memFile.is_open() && !memFile.eof()) memFile >> std::hex >> HEAP_ADDR;
	else throw std::string("mem.txt management error");

	memFile.close();
	system("rm mem.txt");

	if(HEAP_ADDR == 0x0) throw std::string("FCEU mem not found");

	RAM_ADDR = getMemValue(HEAP_ADDR+RAMPTR_ADDR_OFFSET,sizeof(off_t));
	XBUFF_ADDR = getMemValue(HEAP_ADDR+XBUFFPTR_ADDR_OFFSET,sizeof(off_t));

//	std::cout << "PID: "<< pid << "\n";
//
//	std::cout << "MEM ADDR:";
//	std::cout << std::hex << MEM_ADDR;
//	std::cout << "\n";
//
//	std::cout << "RAM PTR ADDR:";
//	std::cout << std::hex << (MEM_ADDR+RAMPTR_ADDR_OFFSET);
//	std::cout << "  RAM ADDR:";
//	std::cout << std::hex << RAM_ADDR;
//	std::cout << "\n";
//
//	std::cout << "XBUFF PTR ADDR:";
//	std::cout << std::hex << (MEM_ADDR+XBUFFPTR_ADDR_OFFSET);
//	std::cout << "  XBUFF ADDR:";
//	std::cout << std::hex << XBUFF_ADDR;
//	std::cout << "\n";
}

/*
 *
 */
MemoryAnalyzer::~MemoryAnalyzer() {

}

/*
 *
 */
off_t MemoryAnalyzer::getMemValue(off_t addr, size_t size = 1)
{

	char file[64];
	sprintf(file, "/proc/%ld/mem", (long)pid);
	int fd = open(file, O_RDWR);
	if(fd == -1) throw std::string("Could not read memory data");

	ptrace(PTRACE_ATTACH, pid, 0, 0);
	waitpid(pid, NULL, 0);

	off_t value = 0;
	pread(fd, &value, size, addr);

	ptrace(PTRACE_DETACH, pid, 0, 0);
	close(fd);

	return value;

}

/*
 *
 */
char MemoryAnalyzer::setMemValue(off_t addr, char value)
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
	if(result.playerVelocityX > 128) result.playerVelocityX = result.playerVelocityX-255;
	if(result.playerVelocityY > 128) result.playerVelocityY = result.playerVelocityY-255;

	result.playerPositionX = (unsigned int) getMemValue(RAM_ADDR+RAM_POS_X_OFFSET);
	result.playerPositionY = (unsigned int) getMemValue(RAM_ADDR+RAM_POS_Y_OFFSET);

	result.screenPosition = (unsigned int) getMemValue(RAM_ADDR+RAM_POS_SCREEN_OFFSET);
	result.screenVelocity = (unsigned int) getMemValue(RAM_ADDR+RAM_VEL_SCREEN_OFFSET);

//	std::cout << result.playerVelocityX << "  " << result.playerVelocityY << "\n";

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
	pread(fd, paletteData, 256*4*sizeof(char), HEAP_ADDR+PALETTE_ADDR);

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


void MemoryAnalyzer::setController(int c)
{
	setMemValue(HEAP_ADDR+CONTROL_ADDR,c);
}

void MemoryAnalyzer::loadState()
{
	setMemValue(HEAP_ADDR+LOADSTATE_ADDR,1);
}
