#include <string>
#include <assert.h>

#include "Bot/Bot.h"
#include "test/NNtestCPU.h"
#include "test/NNtestGPU.h"

/*
 *
 */
int main()
{
	Bot bot;
	try
	{
		bot.execute();
	}
	catch(std::string& e)
	{
		std::cout << "Exception occured: " << e << "\n";
	}
}
