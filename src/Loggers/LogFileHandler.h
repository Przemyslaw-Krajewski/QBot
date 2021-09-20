/*
 * LogFileHandler.h
 *
 *  Created on: 19 wrz 2021
 *      Author: przemo
 */

#ifndef SRC_LOGGERS_LOGFILEHANDLER_H_
#define SRC_LOGGERS_LOGFILEHANDLER_H_

#include<fstream>

class LogFileHandler
{
public:
	static void logValue(const char* t_fileName, double t_value)
	{
		std::ofstream file;
		file.open(t_fileName, std::ios_base::app);
		file << t_value << "\n";
		file.close();
	}
};

#endif /* SRC_LOGGERS_LOGFILEHANDLER_H_ */
