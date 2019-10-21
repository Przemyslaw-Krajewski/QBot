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
	reset = false;
	controlMode = ControlMode::QL;

	/*  Initialize */
	cv::waitKey(3000);
	//Load game in order to avoid not finding player during initializing
//	DesktopHandler::getPtr()->releaseControllerButton();
	MemoryAnalyzer::getPtr()->setController(0);
	MemoryAnalyzer::getPtr()->loadState();

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
	DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
	cv::waitKey(1000);

	//Initialize qLearning
	qLearning = new ActorCritic(numberOfActions, std::vector<int>(sceneState.size(),1));

	playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
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
		double score = 0;

		//State variables
		std::list<SARS> historyScenario;
		State sceneState;
		ControllerInput controllerInput = determineControllerInput(0);
		int action = 0;
		ScenarioResult scenarioResult = ScenarioResult::noInfo;
		int time = TIME_LIMIT;

		//Reload game
//		DesktopHandler::getPtr()->releaseControllerButton();
		MemoryAnalyzer::getPtr()->setController(0);
		MemoryAnalyzer::getPtr()->loadState();
		cv::waitKey(100);

		//Get first state
		while(1)
		{
			StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
			sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
							controllerInput,
							analyzeResult.playerCoords,
							analyzeResult.playerVelocity);
			if(sceneState.size() > 15) break;
			std::cout << "Problem with getting first image\n";
		}

		while(1)
		{
			//Persist info
#ifdef PRINT_PROCESSING_TIME
			int64 timeBefore = cv::getTickCount();
#endif
			cv::waitKey(80);
			std::vector<int> oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyze();
			if(analyzeResult.fieldAndEnemiesLayout.cols == 0 || analyzeResult.fieldAndEnemiesLayout.rows == 0) continue;
			sceneState = createSceneState(analyzeResult.fieldAndEnemiesLayout,
						 	 	 	 	  controllerInput,
										  analyzeResult.playerCoords,
										  analyzeResult.playerVelocity);

			if(controllerInput[0] && analyzeResult.playerVelocity.y == 0 && analyzeResult.reward > 0.01) analyzeResult.reward -= 0.01;
			if(analyzeResult.reward > StateAnalyzer::LITTLE_ADVANCE_REWARD ) score++ ;

			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));

			//Determine new controller input
			action = qLearning->chooseAction(sceneState, controlMode).second;
			controllerInput = determineControllerInput(action);
			MemoryAnalyzer::getPtr()->setController(determineControllerInputInt(action));
//			DesktopHandler::getPtr()->pressControllerButton(controllerInput);

			//Draw info
			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extraxtedSceneData = extractSceneState(sceneState);
			DataDrawer::drawAnalyzedData(extraxtedSceneData.first,extraxtedSceneData.second,
					analyzeResult.reward,0);

#ifdef PRINT_PROCESSING_TIME
			int64 timeAfter = cv::getTickCount();
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
//		DesktopHandler::getPtr()->releaseControllerButton();
		MemoryAnalyzer::getPtr()->setController(0);

		loadParameters();

		//Reset?
		if(reset)
		{
			std::cout << "RESET NN\n";
			qLearning->resetActionsNN();
			reset = false;
		}

		std::cout << score << "\n";

		if(scenarioResult == ScenarioResult::killedByEnemy) eraseInvalidLastStates(historyScenario);
		learnFromScenarioAC(historyScenario);
		learnFromMemoryAC();

//		if(scenarioResult==ScenarioResult::killedByEnemy) eraseInvalidLastStates(historyScenario);
//		if(controlMode != ControlMode::NNNoLearn) learnFromScenarioQL(historyScenario);
//		eraseNotReadyStates();
//
//		if( controlMode == ControlMode::Hybrid || controlMode == ControlMode::NN )
//		{
//			playsBeforeNNLearning--;
//			std::cout << PLAYS_BEFORE_NEURAL_NETWORK_LEARNING-playsBeforeNNLearning << "/" << PLAYS_BEFORE_NEURAL_NETWORK_LEARNING << "\n";
//		}
//		else playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
//
//		if( playsBeforeNNLearning < 1 && (controlMode == ControlMode::Hybrid || controlMode == ControlMode::NN) )
//		{
//			//Learning
//			learnFromMemory();
//			learnFromScenario(historyScenario);
//			playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
//		}
	}
}

/*
 *
 */
void Bot::loadParameters()
{
	std::ifstream qlFile ("ql.param");
	if (qlFile.is_open())
	{
		qlFile.close();
		std::remove("ql.param");
		controlMode = ControlMode::QL;
		std::cout << "QL mode activated\n";
	}

	std::ifstream nnFile ("nn.param");
	if (nnFile.is_open())
	{
		nnFile.close();
		std::remove("nn.param");
		controlMode = ControlMode::NN;
		std::cout << "NN mode activated\n";
	}

	std::ifstream hybridFile ("hybrid.param");
	if (hybridFile.is_open())
	{
		hybridFile.close();
		std::remove("hybrid.param");
		controlMode = ControlMode::Hybrid;
		std::cout << "Hybrid mode activated\n";
		playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
	}

	std::ifstream nnNoLearnFile ("nnnolearn.param");
	if (nnNoLearnFile.is_open())
	{
		nnNoLearnFile.close();
		std::remove("nnnolearn.param");
		controlMode = ControlMode::NNNoLearn;
		std::cout << "NN no learn mode activated\n";
	}

	std::ifstream resetFile ("reset.param");
	if (resetFile.is_open())
	{
		resetFile.close();
		std::remove("reset.param");
		reset = true;
		std::cout << "Reset has been ordered\n";
	}
}

/*
 *
 */
void Bot::learnFromScenarioQL(std::list<SARS> &historyScenario)
{

	//QLearning
	double change = 0;
	long toSkip = 0;
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
	{

		double ch = abs(qLearning->learn(sarsIterator->oldState, sarsIterator->state, sarsIterator->action, sarsIterator->reward + cumulatedReward));
		cumulatedReward = ActorCritic::LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
		if(ch > QLearning::ACTION_LEARN_THRESHOLD)
		{
			toSkip++;
			change += ch;
		}
	}
	std::cout << "To skip: " << toSkip << " History size: " << historyScenario.size() << " Change: " << change << "\n";
}

/*
 *
 */
void Bot::learnFromScenarioAC(std::list<SARS> &historyScenario)
{
	std::vector<SARS*> sarsPointers;
	sarsPointers.clear();
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
	{
		cumulatedReward = ActorCritic::LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
		sarsIterator->reward = cumulatedReward;
		sarsPointers.push_back(&(*sarsIterator));
		memorizedSARS[reduceStateResolution(sarsIterator->oldState, sarsIterator->action)] = SARS(sarsIterator->oldState,
																								  sarsIterator->state,
																								  sarsIterator->action,
																								  sarsIterator->reward);
	}

	std::random_shuffle(sarsPointers.begin(),sarsPointers.end());

	//QLearning
	double sumErr = 0;
	for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
	{
		sumErr += abs(qLearning->learn((*sarsIterator)->oldState,
									   (*sarsIterator)->state,
									   (*sarsIterator)->action,
									   (*sarsIterator)->reward));
	}

//	std::cout << "Cumulated reward: " << cumulatedReward << "\n";
//	std::cout << "History size: " << historyScenario.size() << " sumErr: " << sumErr << "\n";
//	std::cout << "Sum Error: " << sumErr/historyScenario.size() << "\n";
}

/*
 *
 */
void Bot::learnFromMemoryAC()
{
	int skipStep = sqrt(memorizedSARS.size())/8;
	if(skipStep < 1) skipStep = 1;

	double sumErr = 0;
	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return;
	if(memorizedSARS.size() <= 0) return;

	//Prepare states
	std::vector<const SARS*> shuffledSARS;
	for(std::map<ReducedState, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++) shuffledSARS.push_back(&(i->second));

	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
	{

		std::random_shuffle(shuffledSARS.begin(),shuffledSARS.end());
		for(int j=0; j<shuffledSARS.size(); j+=skipStep)
		{
			sumErr += abs(qLearning->learn((shuffledSARS[j])->oldState,
										   (shuffledSARS[j])->state,
										   (shuffledSARS[j])->action,
										   (shuffledSARS[j])->reward));
		}

//		sumErr = 0;
//		for(std::map<ReducedState, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++)
//		{
//			double nnValue = qLearning->getCriticValue(i->second.oldState);
//			sumErr += abs(nnValue - i->second.reward);
//			std::cout << nnValue << "  ->  " << i->second.reward << "\n";
//		}

//		std::cout << "Error mem: " << sumErr / ((double) shuffledSARS.size()) << "\n";
	}
}

/*
 *
 */
void Bot::learnFromScenario(std::list<SARS> &historyScenario)
{

//	if(LEARN_FROM_HISTORY_ITERATIONS == 0) return;
//
//	for(int i=0; i<LEARN_FROM_HISTORY_ITERATIONS; i++)
//	{
//		int skipped = 0;
//		int alreadyOK = 0;
//		double error = 0;
//		for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
//		{
//			if(sarsIterator->action == -1) continue;
//
//			std::pair<double,int> learnResult = qLearning->learnAction(&sarsIterator->oldState, true);
//
//			if( learnResult.second == 2) skipped++; // QL not sure
//			else if(learnResult.second == 3 || learnResult.second == 1) alreadyOK++;
//			else error += learnResult.first;
//
////			if(learnResult.second == 2 || learnResult.second == 3) sarsIterator->action = -1;
//		}
//
//		std::cout << "Error hist: " << error / ((double)historyScenario.size()-skipped-alreadyOK) << "\n";
//		std::cout << "Skipped hist: " << skipped << "/" << alreadyOK << "/" << historyScenario.size() << "\n";
//	}
}

/*
 *
 */
void Bot::learnFromMemory()
{
//	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return;
//	if(discoveredStates.size() <= 0) return;
//
//	//Prepare states
//	std::vector<const State*> shuffledStates;
//	for(std::map<ReducedState, State>::iterator i=discoveredStates.begin(); i!=discoveredStates.end(); i++) shuffledStates.push_back(&(i->second));
//
//	//Learn NN
//	int skipStep = 1;
//	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
//	{
//		int skipped = 0;
//		int alreadyOK = 0;
//		double error = 0;
//		std::random_shuffle(shuffledStates.begin(),shuffledStates.end());
//		for(int j=0; j<shuffledStates.size(); j+=skipStep)
//		{
//
//			std::pair<double,int> learnResult = qLearning->learnAction(&(*(shuffledStates[j])));
//
//			if( learnResult.second == 2) skipped++; // QL not sure
//			else if(learnResult.second == 3 || learnResult.second == 1) alreadyOK++;
//			else error += learnResult.first;
//		}
//
//		std::cout << "Error mem: " << error / ((double) shuffledStates.size()) << "\n";
//		std::cout << "Skipped mem: "<< skipped << "/" << alreadyOK << "/" << (shuffledStates.size()/skipStep) << "\n" ;
//	}
}

/*
 *
 */
void Bot::eraseInvalidLastStates(std::list<SARS> &t_history)
{
	int lastReward = t_history.front().reward;
	std::vector<int> state = t_history.front().oldState;
	int counter = 0;
	while(t_history.size()>0)
	{
		counter++;
		state = t_history.front().oldState;
		if(	state[2659]==0&&state[2660]==0&&state[2661]==0&&state[2662]==0&&state[2715]==0&&state[2771]==0&&state[2718]==0&&state[2774]==0&&
			state[2830]==0&&state[2829]==0&&state[2828]==0&&state[2827]==0&&state[2658]==0&&state[2663]==0&&state[2714]==0&&state[2719]==0&&
			state[2770]==0&&state[2775]==0&&state[2831]==0&&state[2826]==0&&state[2602]==0&&state[2603]==0&&state[2604]==0&&state[2605]==0&&
			state[2606]==0&&state[2607]==0&&state[2882]==0&&state[2883]==0&&state[2884]==0&&state[2885]==0&&state[2886]==0&&state[2887]==0)
			t_history.pop_front();
		else break;
	}
	t_history.front().reward=lastReward;
//	std::cout << "Invaled states: " << counter << "\n";

	std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extractedSceneState = extractSceneState(state);
	DataDrawer::drawAnalyzedData(extractedSceneState.first,extractedSceneState.second,
			lastReward,0);
	if(counter > 5) t_history.clear();
}

/*
 *
 */
void Bot::eraseNotReadyStates()
{
//	long erased = 0;
//	for(std::map<ReducedState, State>::iterator it = discoveredStates.begin(); it!=discoveredStates.end();)
//	{
//		double change = qLearning->getQChange(it->second);
//		if(change > QLearning::ACTION_LEARN_THRESHOLD)
//		{
//			erased++;
//			it = discoveredStates.erase(it);
//		}
//		else it++;
//	}
//
//	std::cout << "Erased: " << erased << " states  =>  " << discoveredStates.size() << "\n";
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
int Bot::determineControllerInputInt(int t_action)
{

	switch(t_action)
	{
	case 0: //Right
		return 128;
	case 1: //Right jump
		return 128+1;
	case 2: //Left
		return 64;
	case 3: //Jump
		return 1;
	case 4: //Left Jump
		return 64+1;
	default:
		std::cout << t_action << "\n";
		assert("No such action!" && false);
	}
	return 0;
}

/*
 *
 */
State Bot::reduceStateResolution(const State& t_state, double action)
{
	int reduceLevel = 2;
	std::vector<int> result;
	for(int i=0;i<t_state.size()-10;i++)
	{
		if(i%reduceLevel!=0 ||( ((int)i/56)%reduceLevel!=0) ) continue;
		result.push_back(t_state[i]);
	}
	for(int i=t_state.size()-10;i<t_state.size();i++)
	{
		result.push_back(t_state[i]);
	}
	result.push_back(action);

	return result;
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
		DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
		std::cout << ": " << analyzeResult.reward << "   " << time << "\n";
		std::cout << analyzeResult.playerVelocity.x << "  " << analyzeResult.playerVelocity.y << "\n";
	}
}
