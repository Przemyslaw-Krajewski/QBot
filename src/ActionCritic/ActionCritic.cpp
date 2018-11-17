/*
 * ActionCritic.cpp
 *
 *  Created on: 23 paź 2018
 *      Author: przemo
 */

#include "ActionCritic.h"

ActionCritic::ActionCritic(int t_nActions, std::vector<int> t_dimensionStatesSize)
{
	alpha = 0.95;
	gamma = 0.9;

	criticValues = new Table(1,t_dimensionStatesSize);
	actorValues = new Table(t_nActions,t_dimensionStatesSize);
	numberOfActions = t_nActions;
}

ActionCritic::~ActionCritic()
{

}

/*
 *
 */
double ActionCritic::learnCritic(State t_prevState, State t_state, int t_action, double t_reward)
{
	//Learn V (critic)
	double value = criticValues->getValue(t_state,0);
	double prevValue = criticValues->getValue(t_prevState,0);

	double newValue = prevValue + alpha*(t_reward+gamma*value - prevValue);
	criticValues->setValue(t_prevState, 0, newValue);

	return value - prevValue;
}

/*
 *
 */
double ActionCritic::learnActor(State t_prevState, State t_state, int t_action, double t_reward)
{
	double value = criticValues->getValue(t_state,0);
	double prevValue = criticValues->getValue(t_prevState,0);

	double maxValue = -999;
	int prevAction;
	for(int i_action=0; i_action<numberOfActions; i_action++)
	{
		double value = actorValues->getValue(t_state,i_action);
		if(maxValue < value)
		{
			maxValue = value;
			prevAction = i_action;
		}
	}

	//learn K handed bandit (actor)
	for(int a=0; a<numberOfActions; a++)
	{
		double change;
		if((value > prevValue && a==t_action) || (value <= prevValue && a!=t_action)) change = 0.1*(numberOfActions-1);
		else change = 0;
		actorValues->setValue(t_prevState,a, actorValues->getValue(t_prevState,a)+change );
	}

	double sum = 0;
	for(int a=0; a<numberOfActions; a++) sum +=actorValues->getValue(t_prevState,a);
	for(int a=0; a<numberOfActions; a++) actorValues->setValue(t_prevState,a, actorValues->getValue(t_prevState,a)/sum );

	maxValue = -999;
	int recentAction;
	for(int i_action=0; i_action<numberOfActions; i_action++)
	{
		double value = actorValues->getValue(t_state,i_action);
		if(maxValue < value)
		{
			maxValue = value;
			recentAction = i_action;
		}
	}

	return value > prevValue && recentAction!=prevAction? 1 : 0;
}

/*
 *
 */
int ActionCritic::chooseAction(State t_state)
{
	double sum = 0;
	for(int a=0; a<numberOfActions; a++) sum +=actorValues->getValue(t_state,a);

	double maxValue = -999;
	int action;
	for(int i_action=0; i_action<numberOfActions; i_action++)
	{
		double value = actorValues->getValue(t_state,i_action);

//		std::cout << value << "  ";
		if(maxValue < value)
		{
			maxValue = value;
			action = i_action;
		}
	}
//	std::cout << " --->  " << action << "\n";
	return action;
}

//###################################################################
/*
 *
 */
void ActionCritic::printCriticMap(State state, std::string string)
{
	int fieldSize = 50;

	cv::Mat image;
	image = cv::Mat(600, 600, CV_8UC3);


	for(int y = 0 ; y < image.rows ; y++)
	{
		uchar* ptr = image.ptr((int)y);
		for(int x = 0 ; x < image.cols*3 ; x++)
		{
			*ptr=0;
			ptr = ptr+1;
		}
	}

	for(int y=0; y<10; y++)
	{
		for(int x=0; x<10; x++)
		{
			state[0] = x;
			state[1] = y;
			double v = criticValues->getValue(state,0);
			for(int yBlock=0; yBlock<=fieldSize; yBlock++)
			{
				for(int xBlock=0; xBlock<=fieldSize; xBlock++)
				{
					uchar* ptr1 = image.ptr((int)y*fieldSize+(int)yBlock)+((int)x*fieldSize+(int)xBlock)*3;
					uchar* ptr2 = ptr1+1;
					uchar* ptr3 = ptr1+2;

					if(v>1) {*ptr1=255;*ptr2=255;*ptr3=255;}
					else if(v<0) {*ptr1=0;*ptr2=0;*ptr3=0;}
					else
					{
						*ptr1=255-255*v;
						*ptr3=0;
						*ptr2=255*v;
					}
				}
			}
		}
	}

	imshow(string, image);
	cv::waitKey(20);
}

/*
 *
 */
void ActionCritic::printActorMap(State state, std::string string)
{
	int fieldSize = 50;

	cv::Mat image;
	image = cv::Mat(600, 600, CV_8UC3);


	for(int y = 0 ; y < image.rows ; y++)
	{
		uchar* ptr = image.ptr((int)y);
		for(int x = 0 ; x < image.cols*3 ; x++)
		{
			*ptr=0;
			ptr = ptr+1;
		}
	}

	for(int y=0; y<10; y++)
	{
		for(int x=0; x<10; x++)
		{
			state[0] = x;
			state[1] = y;
			std::vector<double> v = actorValues->getValues(state);
			int largestI = -1;
			double largestV = -1;
			for(int i=0; i<v.size(); i++)
			{
				if(largestV < v[i])
				{
					largestI = i;
					largestV = v[i];
				}
			}
			for(int yBlock=0; yBlock<=fieldSize; yBlock++)
			{
				for(int xBlock=0; xBlock<=fieldSize; xBlock++)
				{
					uchar* ptr1 = image.ptr((int)y*fieldSize+(int)yBlock)+((int)x*fieldSize+(int)xBlock)*3;
					uchar* ptr2 = ptr1+1;
					uchar* ptr3 = ptr1+2;

					int i;

					if(xBlock>34 && yBlock<34 && yBlock>16) i = 2;
					else if(xBlock<16 && yBlock<34 && yBlock>16) i = 3;
					else if(yBlock>34 && xBlock<34 && xBlock>16) i = 1;
					else if(yBlock<16 && xBlock<34 && xBlock>16) i = 0;
					else if(xBlock<34 && xBlock>16 && yBlock<34 && yBlock>16) i = -1;
					else i = -2;

					*ptr1=0;
					*ptr2=0;
					*ptr3=0;
					if(i >= 0)
					{
						if(v[i] > 1) {*ptr1=255;*ptr2=255;*ptr3=255;}
						else if(i != largestI) {*ptr1=255-255*v[i];*ptr3=255*v[i];}
						else {*ptr2=255*v[i];}
					}
					else if( i == -1) {*ptr1=255; *ptr2=255; *ptr3=255;}
					else {*ptr1=0; *ptr2=0; *ptr3=0;}
				}
			}
		}
	}

	imshow(string, image);
	cv::waitKey(20);
}
