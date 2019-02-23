#include <string>
#include <assert.h>

#include "Analyzers/StateAnalyzer.h"
#include "Flags.h"
#include "QLearning/QLearning.h"
#include "Bot/Bot.h"

void testNN();

/*
 *
 */
int main()
{

#ifdef ENABLE_LOGGING
	freopen( "logs.log", "w", stderr );
#endif
	Bot bot;
	try
	{
		bot.run();
	}
	catch(std::string& e)
	{
		std::cout << "Exception occured: " << e << "\n";
	}
}
