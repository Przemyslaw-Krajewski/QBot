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
	scenarioResult = ScenarioResult::noInfo;
	time = 0;
	persistCounter = 0;
	randomAction = 0;

	/*  Initialize */
	cv::waitKey(3000);
	//Load game in order to avoid not finding player during initializing
	DesktopHandler::getPtr()->releaseControllerButton();
	DesktopHandler::loadGame();

	//push 6 controller inputs
	for(int i=0; i<6; i++) controllerInput.push_back(false);

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

	sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
				 	 	 	 	  controllerInput,
								  analyzeResult.playerCoords,
								  analyzeResult.playerVelocity);
	StateAnalyzer::printAnalyzeData(analyzeResult);
	cv::waitKey(1000);
	//Initialize qLearning
	qLearning = new QLearning(5, std::vector<int>(sceneState.size(),1), hashmap);
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
std::vector<bool> Bot::determineControllerInput(int t_action)
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
		assert("No such action!" && false);
	}

	return w;
}

/*
 *
 */
std::vector<int> Bot::createSceneState(cv::Mat& fieldAndEnemiesLayout, std::vector<bool>& controllerInput, StateAnalyzer::Point& position, StateAnalyzer::Point& velocity)
{
	std::vector<int> sceneState;

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
	sceneState.push_back(position.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(position.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.x > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);
	sceneState.push_back(velocity.y > 0 ? MAX_INPUT_VALUE: MIN_INPUT_VALUE);

	return sceneState;
}

/*
 *
 */
StateAnalyzer::AnalyzeResult Bot::extractSceneState(std::vector<int> sceneData)
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
			if(sceneData[i] == 1) {ptr[0]=100; ptr[1]=100; ptr[2]=100;}
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
			if(sceneData[i] == 1) {ptr[0]=0; ptr[1]=0; ptr[2]=220;}
			i++;
		}
	}

	StateAnalyzer::AnalyzeResult result;
	result.fieldAndEnemiesLayout = fieldAndEnemiesLayout;

	return result;
}

/*
 *
 */
std::vector<int> Bot::reduceStateResolution(std::vector<int> t_state)
{
	std::vector<int> result;
	for(int i=0;i<t_state.size();i++)
	{
		if(i%2==0 || ( ((int)i/32)%2==0) ) continue;
		result.push_back(t_state[i]);
	}
	return result;
}

/*
 *
 */
void Bot::run()
{
	while(1)
	{
		prepareGameBeforeRun();
		scenarioResult = ScenarioResult::noInfo;
		time = TIME_LIMIT;
		action = 0;

		//Get first state
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
		sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
						controllerInput,
						analyzeResult.playerCoords,
						analyzeResult.playerVelocity);
		while(1)
		{
			std::vector<int> oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
			if(analyzeResult.fieldAndEnemiesLayout.cols == 0 || analyzeResult.fieldAndEnemiesLayout.rows == 0)
			{
				std::cout << "OPS\n";
				continue;
			}
			sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
						 	 	 	 	  controllerInput,
										  analyzeResult.playerCoords,
										  analyzeResult.playerVelocity);

			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));

			//Determine new controller input
			if(randomAction <=0)
			{
				if(rand()%150==-1)
				{
					action = rand()%5;
					std::cout << action << "  SOMETHING NEW\n";
					randomAction=20;
				}
				else
				{
					std::pair<int,double> result = qLearning->chooseAction(sceneState);
					if(result.second > 0.4)	action = result.first;
					else action = visitedSARS.getActionWithGreatestChange(sceneState);

				}
			}
			else randomAction--;
			action = 0;

			controllerInput = determineControllerInput(action);
			DesktopHandler::getPtr()->pressControllerButton(controllerInput);

			//Print info
			StateAnalyzer::printAnalyzeData(analyzeResult);
			for(int i=0; i<5; i++) std::cout << (int) (qLearning->getQValue(i)*1000) << " ";
			std::cout << ": " << analyzeResult.reward << "  " << action << "  " << randomAction << "   " << time << "\n";

			//Stop game?
			if(manageScenarioTime(analyzeResult.reward>=0.012)) break;
			if(analyzeResult.endScenario)
			{
				scenarioResult = analyzeResult.additionalInfo;
				break;
			}
		}
		printScenarioResult();
		DesktopHandler::getPtr()->releaseControllerButton();
		learnQLearningScenario();
	}
}

/*
 *
 */
void Bot::prepareGameBeforeRun()
{
	DesktopHandler::getPtr()->releaseControllerButton();
	DesktopHandler::loadGame();
	cv::waitKey(3000);
	historyScenario.clear();
}

/*
 * @return true if is time out
 */
bool Bot::manageScenarioTime(bool resetTimer)
{
	if(!resetTimer)
	{
		time--;
		if(time<0)
		{
			scenarioResult = ScenarioResult::timeOut;
			return true;
		}
		return false;
	}
	else
	{
		time = TIME_LIMIT;
		return false;
	}
}

/*
 *
 */
void Bot::learnQLearningScenario()
{
	//Eliminate invalid last states
	double lastReward = historyScenario.front().reward;
	std::vector<int> state = historyScenario.front().oldState;
	while(historyScenario.size()>0 && scenarioResult==ScenarioResult::killedByEnemy)
	{
		state = historyScenario.front().oldState;
		if(	state[2659]==0&&state[2660]==0&&state[2661]==0&&state[2662]==0&&state[2715]==0&&state[2771]==0&&state[2718]==0&&state[2774]==0&&
			state[2830]==0&&state[2829]==0&&state[2828]==0&&state[2827]==0&&state[2658]==0&&state[2663]==0&&state[2714]==0&&state[2719]==0&&
			state[2770]==0&&state[2775]==0&&state[2831]==0&&state[2826]==0&&state[2602]==0&&state[2603]==0&&state[2604]==0&&state[2605]==0&&
			state[2606]==0&&state[2607]==0&&state[2882]==0&&state[2883]==0&&state[2884]==0&&state[2885]==0&&state[2886]==0&&state[2887]==0)
					historyScenario.pop_front();
		else break;
	}
	if(historyScenario.size()==0) return; //assert("All SARS has been removed" && false);
	historyScenario.front().reward=lastReward;

	StateAnalyzer::AnalyzeResult tmpResult = extractSceneState(state);
	StateAnalyzer::printAnalyzeData(tmpResult);

	long skipNumber;
	//Add learn info
	int counter = 0;
	skipNumber=2;
	for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
	{
		if(counter%skipNumber==0)
			visitedSARS.addSARS(*sarsIterator);
		counter++;
	}

	//Learn
	double change = 0;
	std::cout << "Number of probes:" << historyScenario.size() << "\n";
	skipNumber=3;
	for(int i=0; i<LEARNING_LOOPS; i++)
	{
		long counter = 0;
		for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
		{
			if(counter%skipNumber==0) change += fabs(qLearning->learn(sarsIterator->oldState, sarsIterator->state, sarsIterator->action, sarsIterator->reward));
			counter++;
		}

		//qValues -> target ?
		persistCounter++;
		if(persistCounter >= ITERATIONS_BEFORE_NN_PERSIST)
		{
			persistCounter = 0;
			qLearning->persistNN();
			std::cout << "REWRITE!!\n";
		}
	}
	std::cout << "Change:" << change/historyScenario.size() << "\n";

	//Learn2
	std::cout << "Visited SARS Size: " << visitedSARS.size() << "\n";
	skipNumber=sqrt(visitedSARS.size())/2;
	if(skipNumber < 1) skipNumber = 1;
	for(int i=0; i<LEARNING_LOOPS; i++)
	{
		int counter=0;
		for(VisitedSARS::Iterator it = visitedSARS.begin(); it.hasNextElement(); it++)
		{
			if(counter%skipNumber==0)
			{
				for(int a=0; a<5; a++)
				{
					if(it.existsSars(a))
					{
						SARS s = it.getSars(a);
						double change = qLearning->learn(s.oldState,s.state,s.action,s.reward);
						//it.setChange(a,abs(change));
					}
				}
//			    std::vector<SARS> sars;
//				for(int a=0; a<5; a++)
//				{
//					if(it.existsSars(a)) sars.push_back(it.getSars(a));
//					else sars.push_back(SARS(State(), State(), -1, 0));
//				}
//				qLearning->learn(sars);
			}
			counter++;
		}

		//qValues -> target ?
		persistCounter++;
		if(persistCounter >= ITERATIONS_BEFORE_NN_PERSIST)
		{
			persistCounter = 0;
			qLearning->persistNN();
			std::cout << "REWRITE!!\n";
		}
	}
}

/*
 *
 */
void Bot::printScenarioResult()
{
	//Print death reason
	switch(scenarioResult)
	{
		case ScenarioResult::won:
			std::cout << "Won" << "\n"; break;
		case ScenarioResult::timeOut:
			std::cout << "TimeOut" << "\n"; break;
		case ScenarioResult::killedByEnemy:
			std::cout << "Enemy" << "\n"; break;
		case ScenarioResult::fallenInPitfall:
			std::cout << "Pitfall " << "\n"; break;
		default:
			std::cout << "????" << "\n"; break;
	}
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
		StateAnalyzer::printAnalyzeData(analyzeResult);
		std::cout << ": " << analyzeResult.reward << "   " << time << "\n";
		std::cout << analyzeResult.playerVelocity.x << "  " << analyzeResult.playerVelocity.y << "\n";
	}
}
