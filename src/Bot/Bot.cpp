/*
 * Bot.cpp
 *
 *  Created on: 13 lut 2019
 *      Author: mistrz
 */

#include "Bot.h"

/*
 *
 */
Bot::Bot()
{
	/*  Initialize */
	cv::waitKey(3000);
	//Load game in order to avoid not finding player during initializing
	DesktopHandler::getPtr()->releaseControllerButton();
	DesktopHandler::loadGame();

	//Initialize scene data
	StateAnalyzer::AnalyzeResult analyzeResult;
	for(int i=1; i<11; i++)
	{
		analyzeResult = analyzer.analyze();
		if(analyzeResult.additionalInfo != StateAnalyzer::AnalyzeResult::notFound) break;
		cv::waitKey(1000);
		std::cout << "Could not find player, atteption: " << i << "\n";
	}
	if(analyzeResult.additionalInfo == StateAnalyzer::AnalyzeResult::notFound)
				throw std::string("Could not initialize, check player visibility");

	ControllerInput controllerInput = determineControllerInput(0);
	State sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
								  controllerInput,
								  analyzeResult.playerCoords,
								  analyzeResult.playerVelocity);
	DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0));
	cv::waitKey(1000);

	//Initialize qLearning
	qLearning = new QLearning(numberOfActions, std::vector<int>(sceneState.size(),1));
}

/*
 *
 */
Bot::~Bot()
{
	delete qLearning;
}

/*
 *
 */
void Bot::execute()
{
	while(1)
	{
		//State variables
		std::list<SARS> historyScenario;
		State sceneState;
		ControllerInput controllerInput = determineControllerInput(0);
		int action = 0;
		ScenarioResult scenarioResult = ScenarioResult::noInfo;
		int time = TIME_LIMIT;

		DesktopHandler::getPtr()->releaseControllerButton();
		DesktopHandler::loadGame();
		cv::waitKey(3000);

		//Get first state
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
		sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
						controllerInput,
						analyzeResult.playerCoords,
						analyzeResult.playerVelocity);

		while(1)
		{
			int64 timeBefore = cv::getTickCount();

			std::vector<int> oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
			if(analyzeResult.fieldAndEnemiesLayout.cols == 0 || analyzeResult.fieldAndEnemiesLayout.rows == 0) continue;
			sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
						 	 	 	 	  controllerInput,
										  analyzeResult.playerCoords,
										  analyzeResult.playerVelocity);

			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));
			discoveredStates[reduceStateResolution(sceneState)] = sceneState;

			//Determine new controller input
			action = qLearning->chooseAction(sceneState).second;
			controllerInput = determineControllerInput(action);
			DesktopHandler::getPtr()->pressControllerButton(controllerInput);

			//Draw info
			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extraxtedSceneData = extractSceneState(sceneState);
			DataDrawer::drawAnalyzedData(extraxtedSceneData.first,extraxtedSceneData.second);

			int64 timeAfter = cv::getTickCount();

#ifdef PRINT_PROCESSING_TIME
			std::cout << (timeAfter - timeBefore)/ cv::getTickFrequency() << "\n";
#endif

			//Stop game?
			if(analyzeResult.reward<StateAnalyzer::ADVANCE_REWARD)
			{
				time--;
				if(time<0)
				{
					scenarioResult = ScenarioResult::timeOut;
					break;
				}
			}
			else time = TIME_LIMIT;

			if(analyzeResult.endScenario)
			{
				scenarioResult = analyzeResult.additionalInfo;
				break;
			}
		}
		DesktopHandler::getPtr()->releaseControllerButton();

		//Learning
		std::cout << "\nLEARNING\n";
		if(scenarioResult==ScenarioResult::killedByEnemy) eraseInvalidLastStates(historyScenario);
		learnFromScenario(historyScenario);
		learnFromMemory();
	}
}

/*
 *
 */
void Bot::eraseInvalidLastStates(std::list<SARS> &t_history)
{
	int lastReward = t_history.front().reward;
	std::vector<int> state = t_history.front().oldState;
	while(t_history.size()>0)
	{
		state = t_history.front().oldState;
		if(	state[2659]==0&&state[2660]==0&&state[2661]==0&&state[2662]==0&&state[2715]==0&&state[2771]==0&&state[2718]==0&&state[2774]==0&&
			state[2830]==0&&state[2829]==0&&state[2828]==0&&state[2827]==0&&state[2658]==0&&state[2663]==0&&state[2714]==0&&state[2719]==0&&
			state[2770]==0&&state[2775]==0&&state[2831]==0&&state[2826]==0&&state[2602]==0&&state[2603]==0&&state[2604]==0&&state[2605]==0&&
			state[2606]==0&&state[2607]==0&&state[2882]==0&&state[2883]==0&&state[2884]==0&&state[2885]==0&&state[2886]==0&&state[2887]==0)
			t_history.pop_front();
		else break;
	}
	t_history.front().reward=lastReward;

	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extractedSceneState = extractSceneState(state);
	DataDrawer::drawAnalyzedData(extractedSceneState.first,extractedSceneState.second);
}

/*
 *
 */
void Bot::learnFromScenario(std::list<SARS> &historyScenario)
{
	if(LEARN_FROM_HISTORY_ITERATIONS == 0) return;

	for(int i=0; i<LEARN_FROM_HISTORY_ITERATIONS; i++)
	{
		int skipped = 0;
		int alreadyOK = 0;
		for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
		{
			qLearning->learnQL(sarsIterator->oldState, sarsIterator->state, sarsIterator->action, sarsIterator->reward);
			double learnResult = qLearning->learnAction(sarsIterator->oldState);
			if( learnResult == -1) skipped++;
			else if(learnResult == -2) alreadyOK++;
		}
		std::cout << "Skipped hist: " << skipped << "/" << alreadyOK << "/" << historyScenario.size() << "\n";
	}
}

/*
 *
 */
void Bot::learnFromMemory()
{
	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return;
	std::cout << "Discovered: " << discoveredStates.size() << "\n";
	if(discoveredStates.size() <= 0) return;

	//Prepare states
	std::vector<const State*> shuffledStates;
	for(std::map<State,State>::iterator i=discoveredStates.begin(); i!=discoveredStates.end(); i++) shuffledStates.push_back(&(i->second));

	//Learn NN
	int skipStep = 2;
	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
	{
		int skipped = 0;
		int alreadyOK = 0;
		std::random_shuffle(shuffledStates.begin(),shuffledStates.end());
		for(int j=0; j<shuffledStates.size(); j+=skipStep)
		{
			double learnResult = qLearning->learnAction(*(shuffledStates[j]));
			if( learnResult == -1) skipped++;
			else if(learnResult == -2) alreadyOK++;
		}
		std::cout << "Skipped mem: "<< skipped << "/" << alreadyOK << "/" << (shuffledStates.size()/skipStep) << "\n" ;
	}
}

/*
 *
 */
ControllerInput Bot::determineControllerInput(int t_action)
{
	ControllerInput w;
	for(int i=0; i<numberOfControllerInputs; i++) w.push_back(false);

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
		assert("No such action!" && false);
	}

	return w;
}

/*
 *
 */
std::vector<int> Bot::createSceneState(cv::Mat& fieldAndEnemiesLayout, ControllerInput& controllerInput, Point& position, Point& velocity)
{
	State sceneState;

	//Terrain
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(ptr[0]==100 && ptr[1]==100 && ptr[2]==100) sceneState.push_back(MAX_INPUT_VALUE);
			else sceneState.push_back(MIN_INPUT_VALUE);
		}
	}
	//Enemies
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(ptr[0]==0 && ptr[1]==0 && ptr[2]==220) sceneState.push_back(MAX_INPUT_VALUE);
			else sceneState.push_back(MIN_INPUT_VALUE);
		}
	}
	//Controller
	for(bool ci : controllerInput)
	{
		if(ci==true) sceneState.push_back(MAX_INPUT_VALUE);
		else sceneState.push_back(MIN_INPUT_VALUE);
	}
	//AdditionalInfo
	sceneState.push_back(position.x);
	sceneState.push_back(position.y);
	sceneState.push_back(velocity.x);
	sceneState.push_back(velocity.y);

	return sceneState;
}

/*
 *
 */
std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> Bot::extractSceneState(State sceneState)
{
	cv::Mat fieldAndEnemiesLayout = cv::Mat(56, 32, CV_8UC3);

	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			ptr[0]=0;
			ptr[1]=0;
			ptr[2]=0;
		}
	}

	//Terrain
	long i=0;
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(sceneState[i] == 1) {ptr[0]=100; ptr[1]=100; ptr[2]=100;}
			else {ptr[0]=0; ptr[1]=0; ptr[2]=0;}
			i++;
		}
	}
	//Enemies
	for(int x=0; x<fieldAndEnemiesLayout.cols; x++)
	{
		for(int y=0; y<fieldAndEnemiesLayout.rows; y++)
		{
			uchar* ptr = fieldAndEnemiesLayout.ptr(y)+x*3;
			if(sceneState[i] == 1) {ptr[0]=0; ptr[1]=0; ptr[2]=220;}
			i++;
		}
	}

	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> result ;
	result.first.fieldAndEnemiesLayout = fieldAndEnemiesLayout;

	//AdditionalInfo
	result.first.playerCoords.x = sceneState[sceneState.size()-4];
	result.first.playerCoords.y = sceneState[sceneState.size()-3];
	result.first.playerVelocity.x = sceneState[sceneState.size()-2];
	result.first.playerVelocity.y = sceneState[sceneState.size()-1];

	//Controller
	for(int i=0;i<6; i++) result.second.push_back(sceneState[sceneState.size()-10+i]);

	return result;
}

/*
 *
 */
void Bot::testStateAnalyzer()
{
	while(1)
	{
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
		//Print info
		DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0));
		std::cout << ": " << analyzeResult.reward << "   " << time << "\n";
		std::cout << analyzeResult.playerVelocity.x << "  " << analyzeResult.playerVelocity.y << "\n";
	}
}
