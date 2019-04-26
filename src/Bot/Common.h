/*
 * Common.h
 *
 *  Created on: 13 kwi 2019
 *      Author: mistrz
 */

#ifndef SRC_BOT_COMMON_H_
#define SRC_BOT_COMMON_H_

#include <assert.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>

	using State = std::vector<int>;

	struct SARS
	{
		SARS() // TODO remove this constructor
		{
			state = State();
			oldState = State();
			reward = action = 0;
			change = 1;
		}
		SARS(State t_oldState, State t_state, int t_action, double t_reward)
		{
			state = t_state;
			oldState = t_oldState;
			reward = t_reward;
			action = t_action;
			change = 1;
		}

		State state;
		State oldState;
		int action;
		double reward;
		double change;
	};

	class VisitedSARS
	{
	public:
		VisitedSARS()
		{
			for(int i=0; i<5; i++) sarsMaps.push_back(SARSMap());
		}

		struct Iterator
		{
			VisitedSARS* vsarsPtr;
			int index;

			Iterator(VisitedSARS* ptr, bool begin)
			{
				vsarsPtr = ptr;
				if(begin) index = 0;
				else index = ptr->shuffledStates.size()-1;
			}
			~Iterator()
			{
				vsarsPtr->shuffledStates.clear();
			}

			Iterator& operator++(int)
			{
				index++;
				return *this;
			}
			Iterator& operator--(int)
			{
				index--;
				return *this;
			}

			bool hasNextElement()
			{
				return index < vsarsPtr->shuffledStates.size();
			}

			SARS& getSars(int t_action)
			{
				return (vsarsPtr->sarsMaps)[t_action][*(vsarsPtr->shuffledStates[index])];
			}

			bool existsSars(int t_action)
			{
				return (vsarsPtr->sarsMaps)[t_action].count(*(vsarsPtr->shuffledStates[index])) > 0;
			}

			void setChange(int t_action, double t_change)
			{
				//(vsarsPtr->sarsMaps)[t_action][*(vsarsPtr->shuffledStates[index])].change = t_change;
			}

		};

		void addSARS(SARS sars)
		{
			discoveredStates.insert(reduceStateResolution(sars.oldState));
			sarsMaps[sars.action][reduceStateResolution(sars.oldState)] = SARS(sars.oldState,sars.state,sars.action,sars.reward);
		}

		double getActionWithGreatestChange(State t_state)
		{
			int action;
//			double maxValue = -1;
//			for(int a=0; a<5; a++)
//			{
//				if((sarsMaps)[a].count(reduceStateResolution(t_state)) > 0)
//				{
//					return a;
//				}
//				else if (sarsMaps[a][reduceStateResolution(t_state)].change > maxValue)
//				{
//					action = a;
//					maxValue = sarsMaps[a][reduceStateResolution(t_state)].change;
//				}
//			}

			return action;
		}

		long size()
		{
			return discoveredStates.size();
		}

		Iterator begin()
		{
			assert("Cannot create more than one VSARS iterator" && shuffledStates.size() == 0);
			for(std::set<State>::iterator i=discoveredStates.begin(); i!=discoveredStates.end(); i++)
			{
				const State* s = &(*i);
				shuffledStates.push_back(s);
			}
			std::random_shuffle(shuffledStates.begin(),shuffledStates.end());
			return Iterator(this,true);
		}

	private:
		static State reduceStateResolution(const State& t_state)
		{
			int reduceLevel = 2;
			std::vector<int> result;
			for(int i=0;i<t_state.size();i++)
			{
				if(i%reduceLevel!=0 ||( ((int)i/56)%reduceLevel!=0) ) continue;
				result.push_back(t_state[i]);
			}

//			for(int i=0;i<t_state.size();i++)
//			{
//				std::cout << t_state[i];
//				if((i+1)%56==0) std::cout << "\n";
//			}
//			std::cout << "\n\n";
//			for(int i=0;i<result.size();i++)
//			{
//				std::cout << result[i];
//				if((i+1)%((int)56/reduceLevel)==0) std::cout << "\n";
//			}
//			std::cout << "\n\n";
			return result;
		}

	private:
		std::set<State> discoveredStates;
		using SARSMap = std::map<State,SARS>;
		std::vector<SARSMap> sarsMaps;

		std::vector<const State*> shuffledStates;
	};

#endif /* SRC_BOT_COMMON_H_ */
