/*
 * QLearning.cpp
 *
 *  Created on: 20 cze 2018
 *      Author: przemo
 */

#include "QLearning.h"

/*
 *
 */
QLearning::QLearning(int t_nActions, std::vector<int> t_dimensionStatesSize, ValueMap t_valueMap)
{
	alpha = 0.9;
	gamma = 0.95;

	if(t_valueMap == table) qValues = new Table(t_nActions,t_dimensionStatesSize);
	else if(t_valueMap == hybrid) qValues = new HybridArray(t_nActions,t_dimensionStatesSize);
	else if(t_valueMap == hashmap) qValues = new HashMapArray(t_nActions,t_dimensionStatesSize);
	else qValues = new NeuralNetworkArray(t_nActions,t_dimensionStatesSize);

	numberOfActions = t_nActions;
}

/*
 *
 */
QLearning::~QLearning()
{
	delete qValues;
}


/*
 *
 */
double QLearning::learn(State t_prevState, State t_state, int t_action, double t_reward)
{
	qValues->printInfo();
	double maxValue = -999;
	for(int i_action=0; i_action<numberOfActions; i_action++)
	{
		double value = qValues->getValue(t_state,i_action);
		if(maxValue < value) maxValue = value;
	}

	double prevValue = qValues->getValue(t_prevState,t_action);
	double value = prevValue + alpha*(t_reward+gamma*maxValue - prevValue);
	qValues->setValue(t_prevState, t_action, value);

	return value - prevValue;
}

/*
 *
 */
int QLearning::chooseAction(State t_state)
{
//	std::cout << "Actions: ";
	double maxValue = -999;
	int action;
	for(int i_action=0; i_action<numberOfActions; i_action++)
	{
		double value = qValues->getValue(t_state,i_action);
		if(maxValue < value)
		{
			maxValue = value;
			action = i_action;
		}
//		std::cout << value << "  ";
	}
//	std::cout << "\n";
	return action;
}

/*####################################################################################*/

/*
 *
 */
void QLearning::printValuesMap(State state, std::string string)
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
			std::vector<double> v = qValues->getValues(state);
			for(int i=0; i<v.size(); i++) v[i] = (v[i]+200)/400;
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
