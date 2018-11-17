/*
 * testFunctions.h
 *
 *  Created on: 23 paź 2018
 *      Author: przemo
 */

#ifndef SRC_TESTFUNCTIONS_H_
#define SRC_TESTFUNCTIONS_H_

#include "Game/Game.h"
#include "QLearning/QLearning.h"
#include "ActionCritic/ActionCritic.h"

/*
 *
 */
void testNN()
{
	NeuralNetworkArray nn(1,std::initializer_list<int>{1000});

	cv::Mat image;
	image = cv::Mat(1000, 1000, CV_8UC3);
	unsigned long long iteration = 0;

	while(1)
	{
		iteration++;
		nn.setValue(std::vector<int>{500},0,0.5);
		double v1 = nn.getValue(std::vector<int>{500},0) - 0.5;
		std::cout << v1 << "\n";

		nn.setValue(std::vector<int>{600},0,0.9);
		double v2 = nn.getValue(std::vector<int>{600},0) -0.9;
		std::cout << v2 << "\n";

		nn.setValue(std::vector<int>{800},0,0.5);
		double v3 = nn.getValue(std::vector<int>{800},0) -0.5;
		std::cout << v3 << "\n";

		if((fabs(v1)+fabs(v2)+fabs(v3))/3 < 0.03) break;
//		if((fabs(v1)+fabs(v2))/2 < 0.005) break;

		for(int y = 0 ; y < image.rows ; y++)
		{
			uchar* ptr = image.ptr((int)y);
			for(int x = 0 ; x < image.cols*3 ; x++)
			{
				*ptr=0;
				ptr = ptr+1;
			}
		}

		for(int x = 0; x < 1000; x+=1)
		{
			std::vector<int> input = std::vector<int>{x};
			double y = nn.getValue(input,0);
			uchar* ptr1 = image.ptr((int)((1-y)*999))+((int)x)*3;
			uchar* ptr2 = image.ptr((int)((1-y)*999))+((int)x)*3+1;
			uchar* ptr3 = image.ptr((int)((1-y)*999))+((int)x)*3+2;

			*ptr1=255;
			*ptr2=255;
			*ptr3=255;

			ptr1 = image.ptr((int)((0.5)*1000))+((int)x)*3;
			*ptr1 = 255;
			ptr1 = image.ptr((int)((0.1)*1000))+((int)x)*3;
			*ptr1 = 255;
		}

		imshow("Network", image);
		cv::waitKey(20);
	}
	std::cout << "Done: " << iteration << "\n";
}

/*
 *
 */
void modifyData(QLearning* t_bot)
{
	//Modify/change data
	for(int x=0; x<10; x++)
	{
		for(int y=0; y<10; y++)
		{
			State state;
			state.push_back(x);
			state.push_back(y);
			int action = t_bot->chooseAction(state);
			for(int a=0; a<4; a++)
			{
//				if(y-5 < 3) bot.setQValue(state,a,0.9);
//				if(x-5 < 2 && y-5 < 3) bot.setQValue(state,a,0.9);
//				if(fabs(y-5) < 3 && x-5 < 3) bot.setQValue(state,a,0.9);
//				if(fabs(x-5) < 2 && fabs(y-5) < 3) bot.setQValue(state,a,0.9);
//				if(x < 7 && y < 7 && (x > 2 || y > 2)) bot.setQValue(state,a,0.9);
				if(action == a) t_bot->setQValue(state,a,0.9);
				else t_bot->setQValue(state,a,0.1);
			}
		}
	}
	t_bot->printValuesMap(State{0,0},"Table");
}

/*
 *
 */
void learnNN(QLearning t_bot)
{
	//Prepare data info to learn
	QLearning bot2(4,std::initializer_list<int>{10,10}, neuralNetwork);
	std::vector<std::pair<State,int>> learningData;
	for(int x=0; x<10; x++)
	{
		for(int y=0; y<10; y++)
		{
			for(int a=0; a<4; a++)
			{
				State state;
				state.push_back(x);
				state.push_back(y);
				learningData.push_back(std::pair<State,int>(state,a));
			}
		}
	}

	//Learn NN
	double prevErr=0;
	while(1)
	{
		double err = 0;
		double maxErr = -9999;
		double minErr = 9999;
		std::random_shuffle(learningData.begin(), learningData.end());
		int counter = 0;
		for(std::pair<State,int> ldata: learningData)
		{
			State state = ldata.first;
			int action = ldata.second;
//			if(action != 0) continue;
			double partialError = fabs(t_bot.getQValue(state,action) - bot2.getQValue(state,action));
			err += partialError;
			if(maxErr < partialError) maxErr = partialError;
			if(minErr > partialError) minErr = partialError;

			bot2.setQValue(state,action,t_bot.getQValue(state,action));

			counter++;
			if(counter > 10)
			{
				bot2.printValuesMap(state,"NN");
				counter = 0;
			}
		}

		std::cout << " -> " << err/learningData.size() << "   " << (prevErr-err)/learningData.size() << "\n";
		std::cout << "    " << minErr << " - " << maxErr << "\n";
//		if(fabs(prevErr-err)/learningData.size() < 0.001) break;
		prevErr = err;

		if( err < 0.06 ) bot2.printArrayInfo();

	}

}

#endif /* SRC_TESTFUNCTIONS_H_ */
