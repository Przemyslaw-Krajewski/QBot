#include <string>
#include <assert.h>

#include "Analyzers/StateAnalyzer.h"
#include "QLearning/QLearning.h"
#include "Bot/Bot.h"
#include "test/NNtest.h"

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