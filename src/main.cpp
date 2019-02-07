#include <string>
#include <assert.h>

#include "Flags.h"
#include "ImageAnalyzer/ImageAnalyzer.h"
#include "QLearning/QLearning.h"

enum DeathReason {unknown, timeOut, enemy, pitfall};

const int MAX_INPUT_VALUE = 1;
const int MIN_INPUT_VALUE= 0;

const int TIME_LIMIT = 150;

void testNN();

//Objects
ImageAnalyzer analyzer;

//Situation data
std::vector<bool> controllerInput;

DeathReason deathReason;

std::vector<int> copySceneState(cv::Mat& image, std::vector<bool>& controllerInput, cv::Point& position, cv::Point& velocity)
{
	std::vector<int> sceneState;
	for(int y=0; y<image.rows; y++)
	{
		for(int x=0; x<image.cols; x++)
		{
			uchar* ptr = image.ptr(y)+x*3;

			if(ptr[0]==0 && ptr[1]==0 && ptr[2]==220) sceneState.push_back(MAX_INPUT_VALUE);
			else sceneState.push_back(MIN_INPUT_VALUE);
			if(ptr[0]==100 && ptr[1]==100 && ptr[2]==100) sceneState.push_back(MAX_INPUT_VALUE);
			else sceneState.push_back(MIN_INPUT_VALUE);
		}
	}
	for(bool ci : controllerInput)
	{
		if(ci==true) sceneState.push_back(MAX_INPUT_VALUE);
		else sceneState.push_back(MIN_INPUT_VALUE);
	}
	sceneState.push_back(position.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(position.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);

	return sceneState;
}

std::vector<int> enrichSceneState(std::vector<int>& sceneState, std::vector<bool>& controllerInput, cv::Point& position, cv::Point& velocity)
{
	for(bool ci : controllerInput)
	{
		if(ci==true) sceneState.push_back(MAX_INPUT_VALUE);
		else sceneState.push_back(MIN_INPUT_VALUE);
	}
	sceneState.push_back(position.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(position.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
//	sceneData.push_back(-1);
//	sceneData.push_back(-1);
//	sceneData.push_back(-1);
//	sceneData.push_back(-1);

	return sceneState;
}

void printAction(int t_action)
{
	switch(t_action)
	{
	case 0: //Right
		std::cout << ">\n";
		break;
	case 1: //Right jump
		std::cout << "^>\n";
		break;
	case 2: //Left
		std::cout << "<\n";
		break;
	case 3: //Jump
		std::cout << "^\n";
		break;
	case 4: //Left Jump
		std::cout << "<^\n";
		break;
	default:
		std::cout << t_action << "\n";
		assert("No such action!");
	}
}

std::vector<bool> determineControllerInput(int t_action)
{
	std::vector<bool> w;
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);
	w.push_back(false);

	switch(t_action)
	{
	case 0: //Right
		w[4] = true;
		break;
	case 1: //Right jump
		w[0] = true;
		w[4] = true;
		break;
	case 2: //Left
		w[2] = true;
		break;
	case 3: //Jump
		w[0] = true;
		break;
	case 4: //Left Jump
		w[0] = true;
		w[2] = true;
		break;
	default:
		std::cout << t_action << "\n";
		assert("No such action!");
	}

	return w;
}

struct HistoryEntry
{
	HistoryEntry(State t_oldState, State t_state, int t_action, double t_reward)
	{
		state = t_state;
		oldState = t_oldState;
		reward = t_reward;
		action = t_action;
	}

	State state;
	State oldState;
	int action;
	double reward;
};

/*
 *
 */
int main()
{
//	testNN();
//	return 0;

	freopen( "logs.log", "w", stderr );
	try
	{
		std::vector<int> sceneState;
		int action = 0;
		//QLearning bot;

		/*  Initialize */
		cv::waitKey(3000);
		//Load game in order to avoid not finding player during initializing
		DesktopHandler::getPtr()->releaseControllerButton();
		DesktopHandler::loadGame();

		//push 6 controller inputs
		for(int i=0; i<6; i++) controllerInput.push_back(false);

		//Initialize scene data
		ImageAnalyzer::AnalyzeResult analyzeResult;
		for(int i=1; i<11; i++)
		{
			analyzeResult = analyzer.processImage();
			if(analyzeResult.additionalInfo != ImageAnalyzer::AnalyzeResult::notFound) break;
			cv::waitKey(1000);
			std::cout << "Could not find player, atteption: " << i << "\n";
		}
		if(analyzeResult.additionalInfo == ImageAnalyzer::AnalyzeResult::notFound)
					throw std::string("Could not initialize, check player visibility");
		sceneState = analyzeResult.data;
		enrichSceneState(sceneState,
						controllerInput,
						analyzeResult.playerCoords,
						analyzeResult.playerVelocity);
		ImageAnalyzer::printAnalyzeData(sceneState, true);
		cv::waitKey(1000);

		//Initialize bot
		QLearning bot = QLearning(5, std::vector<int>(sceneState.size(),1), hashmap);

		while(1)
		{
			DesktopHandler::getPtr()->releaseControllerButton();
			DesktopHandler::loadGame();
			cv::waitKey(3000);

			std::list<HistoryEntry> historyScenario;

			deathReason = unknown;
			int time = TIME_LIMIT;

			//Analyze situation
			ImageAnalyzer::AnalyzeResult analyzeResult = analyzer.processImage();
			sceneState = analyzeResult.data;
			enrichSceneState(sceneState,
							controllerInput,
							analyzeResult.playerCoords,
							analyzeResult.playerVelocity);
			action = 0;

			while(1)
			{
				std::vector<int> oldSceneState = sceneState;
				int oldAction = action;

				//Analyze situation
				ImageAnalyzer::AnalyzeResult analyzeResult = analyzer.processImage();
				sceneState = analyzeResult.data;
				enrichSceneState(sceneState,
								controllerInput,
								analyzeResult.playerCoords,
								analyzeResult.playerVelocity);
				double reward = analyzeResult.reward;
				//If player disappeard then gameplay is stopped
				if(analyzeResult.additionalInfo == ImageAnalyzer::AnalyzeResult::notFound) reward = -100;

				ImageAnalyzer::printAnalyzeData(sceneState, true);

				//add learning info to history
				historyScenario.push_front(HistoryEntry(oldSceneState, sceneState, oldAction, reward));

				if(sceneState.size() == 906) bot.addDiscoveredState(sceneState);
//				else throw "Bad size";

				//Determine new controller input
				std::pair<bool,int> decision = bot.chooseAction(sceneState);
				action = decision.second;
				controllerInput = determineControllerInput(action);
				DesktopHandler::getPtr()->pressControllerButton(controllerInput);

				for(int i=0; i<5; i++) std::cout << (int) bot.getQValue(oldSceneState,i) << " ";
				std::cout << ": " << reward << "   " << time << "\n";

				//Stop game?
				if(reward != 50)
				{
					time--;
					if(time<0)
					{
						deathReason = timeOut;
						break;
					}
				}
				else time = TIME_LIMIT;
				if(reward == -100)
				{
					historyScenario.push_front(HistoryEntry(oldSceneState, sceneState, oldAction, reward));

					if(analyzeResult.additionalInfo == ImageAnalyzer::AnalyzeResult::fallenInPitfall) deathReason = pitfall;
					else deathReason = enemy;

					break;
				}
			}

			//Learn bot
			double change = 0;
			long numberOfProbes = 0;
			for(std::list<HistoryEntry>::iterator historyIterator = historyScenario.begin(); historyIterator!=historyScenario.end(); historyIterator++)
			{
				change += fabs(bot.learn(historyIterator->oldState, historyIterator->state, historyIterator->action, historyIterator->reward));
				numberOfProbes++;
			}

			//Print death reason
			switch(deathReason)
			{
				case timeOut:
					std::cout << "TimeOut" << "\n"; break;
				case enemy:
					std::cout << "Enemy" << "\n"; break;
				case pitfall:
					std::cout << "Pitfall " << "\n"; break;
				default:
					std::cout << "????" << "\n"; break;
			}

			//Summarize scenario
			std::cout << "Change:" << change/numberOfProbes << "\n";
			DesktopHandler::getPtr()->releaseControllerButton();
		}
	}
	catch(std::string e)
	{
		std::cout << "Exception occured: " << e << "\n";
	}
}
