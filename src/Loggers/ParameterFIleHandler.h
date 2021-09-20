/*
 * ParameterFIleHandler.h
 *
 *  Created on: 19 wrz 2021
 *      Author: przemo
 */

#ifndef SRC_BOT_PARAMETERFILEHANDLER_H_
#define SRC_BOT_PARAMETERFILEHANDLER_H_

#include<iostream>
#include<fstream>

class ParameterFileHandler
{
public:
	static bool checkParameter(const char* t_parameterName, std::string t_communiaction)
	{
		std::ifstream file (t_parameterName);
		if (file.is_open())
		{
			file.close();
			std::remove(t_parameterName);
			std::cout << "Parameter handler: " << t_communiaction << "\n";
			return true;
		}
		return false;
	}
};

#endif /* SRC_BOT_PARAMETERFILEHANDLER_H_ */
