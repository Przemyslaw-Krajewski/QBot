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
	MemoryAnalyzer::getPtr()->setController(0);
	MemoryAnalyzer::getPtr()->loadState();

	//Initialize scene data
	StateAnalyzer::AnalyzeResult analyzeResult;

	analyzeResult = analyzer.analyzeBT();
	ControllerInput controllerInput = determineControllerInput(0);
	State sceneState = createSceneState(analyzeResult.processedImage,
								  analyzeResult.processedImagePast,
								  analyzeResult.processedImagePast2,
								  controllerInput,
								  analyzeResult.playerCoords,
								  analyzeResult.playerVelocity);
//	DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
	cv::waitKey(1000);

	//Initialize acLearning
	actorCritic = new ActorCritic(numberOfActions,
	        std::vector<int>(analyzeResult.processedImage.cols*analyzeResult.processedImage.rows*2*3,255));

	playsBeforeNNLearning = PLAYS_BEFORE_NEURAL_NETWORK_LEARNING;
}

/*
 *
 */
Bot::~Bot()
{
	delete actorCritic;
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
//		int time = TIME_LIMIT;


		//Reload game
		MemoryAnalyzer::getPtr()->setController(0);
		MemoryAnalyzer::getPtr()->loadState();
		cv::waitKey(50);

		//Get first state
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyzeBT();
		sceneState = createSceneState(analyzeResult.processedImage,
						analyzeResult.processedImagePast,
						analyzeResult.processedImagePast2,
						controllerInput,
						analyzeResult.playerCoords,
						analyzeResult.playerVelocity);

		while(1)
		{
			//Persist info
#ifdef PRINT_PROCESSING_TIME
			int64 timeBefore = cv::getTickCount();
#endif
			cv::waitKey(40);
			std::vector<int> oldSceneState = sceneState;
			int oldAction = action;

			//Analyze situation
			StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyzeBT();
//			if(analyzeResult.fieldAndEnemiesLayout.cols == 0 || analyzeResult.fieldAndEnemiesLayout.rows == 0) continue;
			sceneState = createSceneState(analyzeResult.processedImage,
										  analyzeResult.processedImagePast,
										  analyzeResult.processedImagePast2,
						 	 	 	 	  controllerInput,
										  analyzeResult.playerCoords,
										  analyzeResult.playerVelocity);

			if(analyzeResult.reward >= StateAnalyzer::LITTLE_ADVANCE_REWARD ) score++ ;
			//add learning info to history
			historyScenario.push_front(SARS(oldSceneState, sceneState, oldAction, analyzeResult.reward));

			//Determine new controller input
			action = actorCritic->chooseAction(sceneState, controlMode).second;
			controllerInput = determineControllerInput(action);
			MemoryAnalyzer::getPtr()->setController(determineControllerInputInt(action));

			//Draw info
//			std::pair<StateAnalyzer::AnalyzeResult, ControllerInput> extraxtedSceneData = extractSceneState(sceneState);
//			DataDrawer::drawAnalyzedData(extraxtedSceneData.first,extraxtedSceneData.second,
//					analyzeResult.reward,actorCritic->getQChange(sceneState));

#ifdef PRINT_PROCESSING_TIME
			int64 afterBefore = cv::getTickCount();
			std::cout << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
#endif
			//End?
			if(analyzeResult.endScenario)
			{
				scenarioResult = analyzeResult.additionalInfo;
				break;
			}
		}

		MemoryAnalyzer::getPtr()->setController(0);

		loadParameters();

		//Reset?
		if(reset)
		{
			std::cout << "RESET NN\n";
            actorCritic->resetNN();
			reset = false;
		}

		std::cout << score << "\n";

		learnFromScenarioAC(historyScenario);
//		learnFromMemoryAC();
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
void Bot::learnFromScenarioAC(std::list<SARS> &historyScenario)
{
	std::vector<SARS*> sarsPointers;
	sarsPointers.clear();
	double cumulatedReward = 0;
	for(std::list<SARS>::iterator sarsIterator = historyScenario.begin(); sarsIterator!=historyScenario.end(); sarsIterator++)
	{
		cumulatedReward = ActorCritic::LAMBDA_PARAMETER*(sarsIterator->reward + cumulatedReward);
		//std::cout << cumulatedReward << "\n"; 
		sarsIterator->reward = cumulatedReward;
		sarsPointers.push_back(&(*sarsIterator));
		memorizedSARS[reduceSceneState(sarsIterator->oldState, sarsIterator->action)] = SARS(sarsIterator->oldState,
																								  sarsIterator->state,
																								  sarsIterator->action,
																								  sarsIterator->reward);
	}

	long counter=0;
	//while(1)
	//{

	std::random_shuffle(sarsPointers.begin(),sarsPointers.end());

	//QLearning
	double sumErr = 0;
	for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
	{
		sumErr += abs(actorCritic->learn((*sarsIterator)->oldState,
										 (*sarsIterator)->state,
										 (*sarsIterator)->action,
										 (*sarsIterator)->reward));
	}
	std::cout << sumErr/sarsPointers.size() << "\n";

	//sumErr = 0;
	//counter++;
	//for(std::vector<SARS*>::iterator sarsIterator = sarsPointers.begin(); sarsIterator!=sarsPointers.end(); sarsIterator++)
	//{
	//	double value = actorCritic->getCriticValue((*sarsIterator)->oldState);
	//	sumErr += abs(value-(*sarsIterator)->reward);
	//	std::cout << value << " = "<< (*sarsIterator)->reward << "\n";
	//}
	//std::cout << sumErr/sarsPointers.size() << "  " << sarsPointers.size() << "  " << counter << "\n";
//	if(counter%20==0) actorCritic->drawCriticValues();
	//	}
}

/*
 *
 */
void Bot::learnFromMemoryAC()
{
	int skipStep = memorizedSARS.size()/150;
	if(skipStep < 1) skipStep = 1;

	double sumErr = 0;
	if(LEARN_FROM_MEMORY_ITERATIONS == 0) return;
	if(memorizedSARS.size() <= 0) return;

	//Prepare states
	std::vector<SARS*> shuffledSARS;
	for(std::map<ReducedState, SARS>::iterator i=memorizedSARS.begin(); i!=memorizedSARS.end(); i++) shuffledSARS.push_back(&(i->second));

	for(int iteration=0; iteration<LEARN_FROM_MEMORY_ITERATIONS; iteration++)
	{

		std::random_shuffle(shuffledSARS.begin(),shuffledSARS.end());
		for(int j=0; j<shuffledSARS.size(); j+=skipStep)
		{
			sumErr += abs(actorCritic->learn((shuffledSARS[j])->oldState,
										     (shuffledSARS[j])->state,
										     (shuffledSARS[j])->action,
										     (shuffledSARS[j])->reward));
		}
	}
}

/*
 *
 */
ControllerInput Bot::determineControllerInput(int t_action)
{
	ControllerInput w;
	for(int i=0; i<numberOfControllerInputs; i++) w.push_back(false);

	w[ 2+(t_action%4) ] = true;
	w[0] = t_action>3;

	return w;
}

/*
 *
 */
int Bot::determineControllerInputInt(int t_action)
{
	int direction = (1<<(4+t_action%4));
	int jump = t_action>3?1:0;
	return direction+jump;
}

/*
 *
 */
State Bot::reduceSceneState(const State& t_state, double action)
{
	int reduceLevel = 4;
	std::vector<int> result;
	for(int i=0;i<t_state.size()-10;i++)
	{
		if(i%reduceLevel!=0) continue;
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
std::vector<int> Bot::createSceneState(cv::Mat& image, cv::Mat& imagePast, cv::Mat& imagePast2,
		ControllerInput& controllerInput, Point& position, Point& velocity)
{
	State sceneState;

	for(int x=0; x<image.cols; x++)
	{
		for(int y=0; y<image.rows; y++)
		{
			uchar* ptrSrc = image.ptr(y)+(3*(x));
//			int c1 = (ptrSrc[0] >> 6);
//			int c2 = (ptrSrc[1] >> 4);
//			int c3 = (ptrSrc[2] >> 2);
//			sceneState.push_back(c1 + c2 + c3);
			sceneState.push_back(ptrSrc[0]);
			sceneState.push_back(ptrSrc[1]);
			sceneState.push_back(ptrSrc[2]);
		}
	}

	for(int x=0; x<imagePast.cols; x++)
	{
		for(int y=0; y<imagePast.rows; y++)
		{
			uchar* ptrSrc = imagePast.ptr(y)+(3*(x));
//			int c1 = (ptrSrc[0] >> 6);
//			int c2 = (ptrSrc[1] >> 4);
//			int c3 = (ptrSrc[2] >> 2);
//			sceneState.push_back(c1 + c2 + c3);
			sceneState.push_back(ptrSrc[0]);
			sceneState.push_back(ptrSrc[1]);
			sceneState.push_back(ptrSrc[2]);
		}
	}

//	for(int x=0; x<imagePast2.cols; x++)
//	{
//		for(int y=0; y<imagePast2.rows; y++)
//		{
//			uchar* ptrSrc = imagePast2.ptr(y)+(3*(x));
//			sceneState.push_back((ptrSrc[0] >> 7) + (ptrSrc[1] >> 6) + (ptrSrc[2] >> 5));
//		}
//	}

	//Controller
//	for(bool ci : controllerInput)
//	{
//		if(ci==true) sceneState.push_back(MAX_INPUT_VALUE);
//		else sceneState.push_back(MIN_INPUT_VALUE);
//	}
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
//	result.first.fieldAndEnemiesLayout = fieldAndEnemiesLayout;
	//TODO disabled

//	//AdditionalInfo
//	result.first.playerCoords.x = sceneState[sceneState.size()-4];
//	result.first.playerCoords.y = sceneState[sceneState.size()-3];
//	result.first.playerVelocity.x = sceneState[sceneState.size()-2];
//	result.first.playerVelocity.y = sceneState[sceneState.size()-1];

	//Controller
	for(int i=0;i<6; i++) result.second.push_back(sceneState[sceneState.size()-1+i]);

	return result;
}

/*
 *
 */
void Bot::testStateAnalyzer()
{
	while(1)
	{
		StateAnalyzer::AnalyzeResult analyzeResult = analyzer.analyzeBT();
		//Print info
//		DataDrawer::drawAnalyzedData(analyzeResult,determineControllerInput(0),0,0);
		std::cout << ": " << analyzeResult.reward << "\n";
	}
}
