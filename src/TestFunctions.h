/*
 * testFunctions.h
 *
 *  Created on: 23 paź 2018
 *      Author: przemo
 */

#ifndef SRC_TESTFUNCTIONS_H_
#define SRC_TESTFUNCTIONS_H_

#include "QLearning/QLearning.h"

/*
 *
 */
void testNN()
{
	int inputSize = 1;
	std::vector<int> layers = std::vector<int>{10,1};
	std::vector<double> n = std::vector<double>{3,1};
	double b = 1.3;
	NeuralNetwork nn(inputSize,layers,n,b);

	cv::Mat image;
	image = cv::Mat(1000, 1000, CV_8UC3);
	unsigned long long iteration = 0;

	while(1)
	{
		iteration++;
		nn.determineY(std::initializer_list<double>({0.4}));
		nn.learnBackPropagation(std::initializer_list<double>({0.2}));
		double v1 = nn.determineY(std::initializer_list<double>({0.4}))[0] - 0.2;
		std::cout << v1 << "\n";

		nn.determineY(std::initializer_list<double>({0.6}));
		nn.learnBackPropagation(std::initializer_list<double>({0.9}));
		double v2 = nn.determineY(std::initializer_list<double>({0.6}))[0] -0.9;
		std::cout << v2 << "\n";

		nn.determineY(std::initializer_list<double>({0.8}));
		nn.learnBackPropagation(std::initializer_list<double>({0.2}));
		double v3 = nn.determineY(std::initializer_list<double>({0.8}))[0] -0.2;
		std::cout << v3 << "\n";

		if((fabs(v1)+fabs(v2)+fabs(v3))/3 < 0.03) break;
//		if((fabs(v1)+fabs(v3))/2 < 0.005) break;

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
			std::vector<double> input = std::vector<double>{(double)x/1000};
			double y = nn.determineY(input)[0];
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
	NeuralNetwork newNN(nn);
	std::cout << "Done: " << iteration << "\n";

	for(int x = 0; x < 1000; x+=2)
	{
		std::vector<double> input = std::vector<double>{(double)x/1000};
		double y = newNN.determineY(input)[0];
		uchar* ptr1 = image.ptr((int)((1-y)*999))+((int)x)*3;
		uchar* ptr2 = image.ptr((int)((1-y)*999))+((int)x)*3+1;
		uchar* ptr3 = image.ptr((int)((1-y)*999))+((int)x)*3+2;

		*ptr1=255;
		*ptr2=255;
		*ptr3=0;
	}

	imshow("Network", image);
	cv::waitKey(200000);
}

#endif /* SRC_TESTFUNCTIONS_H_ */
