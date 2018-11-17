#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "Game/Game.h"
#include "QLearning/QLearning.h"
#include "ActionCritic/ActionCritic.h"

#include "testFunctions.h"

struct HistoryRecord
{
	HistoryRecord(State t_prevState, State t_state, int t_action, double t_result)
	{
		prevState = t_prevState;
		state = t_state;
		action = t_action;
		result = t_result;
	}
	State prevState;
	State state;
	int action;
	double result;
};

void qLearning(Game* t_game, QLearning* t_bot)
{
	std::vector<HistoryRecord> history;
	history.clear();

	while(true)
	{
		State prevState = t_game->getState();
		Control control = Control(t_bot->chooseAction(State(prevState)));
		std::pair<double,bool> result = t_game->execute(control);
		State state = t_game->getState();

		HistoryRecord historyRecord(prevState, state, control.getInt(), result.first);
		history.push_back(historyRecord);
		t_bot->learn(historyRecord.prevState,historyRecord.state,historyRecord.action,historyRecord.result);

		if(result.second)
		{
			double err = 0;
			for(int i=history.size()-1; i>=0; i--)
			{
				err += fabs(t_bot->learn(history[i].prevState,history[i].state,history[i].action,history[i].result));
			}
			std::cout << err/history.size() << "\n";
			history.clear();
			t_game->reset();
		}
		t_bot->printValuesMap(state,"Table");

		cv::waitKey(20);
		t_game->display();
	}
}

void acLearning(Game* t_game, ActionCritic* t_bot)
{
	std::vector<HistoryRecord> history;
	history.clear();

	while(true)
	{
		// Critic
		while(true)
		{
			State prevState = t_game->getState();
			Control control = Control(t_bot->chooseAction(State(prevState)));
			std::pair<double,bool> result = t_game->execute(control);
			State state = t_game->getState();

			HistoryRecord historyRecord(prevState, state, control.getInt(), result.first);
			history.push_back(historyRecord);
//			t_bot->learnCritic(historyRecord.prevState,historyRecord.state,historyRecord.action,historyRecord.result);

			if(result.second)
			{
				double err = 0;
				for(int i=history.size()-1; i>=0; i--)
				{
					err += fabs(t_bot->learnCritic(history[i].prevState,history[i].state,history[i].action,history[i].result));
				}
				std::cout << err/history.size() << "\n";
				int historySize = history.size();
				history.clear();
				t_game->reset();
				if(err/historySize < 0.002)
				{
					std::cout << "Critic done\n";
					break;
				}
			}
			t_bot->printCriticMap(state,"Critic");

			cv::waitKey(20);
			t_game->display();
		}
		//	Actor
		while(true)
		{
			State prevState = t_game->getState();
			Control control = Control(t_bot->chooseAction(State(prevState)));
			std::pair<double,bool> result = t_game->execute(control);
			State state = t_game->getState();

			HistoryRecord historyRecord(prevState, state, control.getInt(), result.first);
			history.push_back(historyRecord);
			t_bot->learnActor(historyRecord.prevState,historyRecord.state,historyRecord.action,historyRecord.result);

			if(result.second)
			{
				double err = 0;
				for(int i=history.size()-1; i>=0; i--)
				{
					err += fabs(t_bot->learnActor(history[i].prevState,history[i].state,history[i].action,history[i].result));
				}
				std::cout << err/history.size() << "\n";
				int historySize = history.size();
				history.clear();
				t_game->reset();
				if(err/historySize < 0.05)
				{
					std::cout << "Actor done\n";
					break;
				}
			}
			t_bot->printActorMap(state,"Actor");

			cv::waitKey(20);
			t_game->display();
		}
	}
}

/*
 *
 */
int main()
{
//	testNN();
//	return 0;

	srand( unsigned ( std::time(0)));

	Game game(10,10);
	QLearning bot(4,std::initializer_list<int>{10,10,10,10},table);
	//QLearning bot(4,game.getState());

	std::vector<std::vector<bool>> level;
	level.push_back(std::vector<bool>{0,0,0,0,0,0,0,0,0,0});
	level.push_back(std::vector<bool>{0,0,0,0,0,0,0,0,0,0});
	level.push_back(std::vector<bool>{0,0,1,1,1,1,1,1,0,0});
	level.push_back(std::vector<bool>{0,0,1,0,0,0,0,1,0,0});
	level.push_back(std::vector<bool>{0,0,1,0,0,0,0,1,0,0});
	level.push_back(std::vector<bool>{0,0,1,0,0,0,0,1,0,0});
	level.push_back(std::vector<bool>{0,0,1,0,0,0,0,1,0,0});
	level.push_back(std::vector<bool>{0,0,1,1,1,0,1,1,0,0});
	level.push_back(std::vector<bool>{0,0,0,0,0,0,0,0,0,0});
	level.push_back(std::vector<bool>{0,0,0,0,0,0,0,0,0,0});
	game.setLevel(level);

	int i = 0;

	std::vector<HistoryRecord> history;
	history.clear();

	qLearning(&game,&bot);

	std::cout << "Toasty\n";
	return 0;
}
