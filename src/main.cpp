#include <string>
#include <assert.h>

#include "Analyzers/StateAnalyzer.h"
#include "QLearning/QLearning.h"
#include "Bot/Bot.h"

void testNN();

/*
 *
 */
int main()
{
	Bot bot;
	try
	{
		bot.execute();
//		bot.testStateAnalyzer();
	}
	catch(std::string& e)
	{
		std::cout << "Exception occured: " << e << "\n";
	}
}
