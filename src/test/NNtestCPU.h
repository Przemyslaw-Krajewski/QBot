//
// Created by przemo on 02.01.2020.
//

#ifndef QBOT_NNTEST_H
#define QBOT_NNTEST_H

#include "../NeuralNetworkGPU/NeuralNetwork.h"
#include "../NeuralNetwork/NeuralNetwork.h"

namespace Test
{

    /*
     *
     */
    long testNeuralNetwork(std::vector<double> t_x, std::vector<double> t_z, NeuralNetworkCPU::NeuralNetwork *nn)
    {
        std::vector<std::vector<double>> x;
        for (double &i : t_x) x.push_back({i});
        std::vector<std::vector<double>> z;
        for (double &i : t_z) z.push_back({i});

        cv::Mat image;
        image = cv::Mat(1000, 1000, CV_8UC3);
        unsigned long long iteration = 0;

        while (1)
        {
            double missSum = 0;
            iteration++;
            std::cout << iteration << "\n";
            for (int i = 0; i < x.size(); i++)
            {
            	double y = (nn->determineOutput(x[i]))[0];
                double miss = y - z[i][0];
                missSum += fabs(miss);
                std::cout << x[i][0] << "  " << y << " -> " << z[i][0] << " Miss: "<< miss << "\n";
            }
            for (int i = 0; i < x.size(); i++)
            {
                nn->determineOutput(x[i]);
                nn->learnBackPropagation(z[i]);
            }

            if(iteration %100 == 0)
            {
				for (int y = 0; y < image.rows; y++)
				{
					uchar *ptr = image.ptr((int) y);
					for (int x = 0; x < image.cols * 3; x++)
					{
						*ptr = 0;
						ptr = ptr + 1;
					}
				}

				for (int x = 0; x < 1000; x += 3)
				{
					std::vector<double> input = std::vector<double>{((double) x) / 1000};
					double y = nn->determineOutput(input)[0];
					uchar *ptr1 = image.ptr((int) ((1 - y) * 999)) + ((int) x) * 3;
					uchar *ptr2 = image.ptr((int) ((1 - y) * 999)) + ((int) x) * 3 + 1;
					uchar *ptr3 = image.ptr((int) ((1 - y) * 999)) + ((int) x) * 3 + 2;

					*ptr1 = 255;
					*ptr2 = 255;
					*ptr3 = 255;

	                ptr1 = image.ptr((int) ((0.6) * 1000)) + ((int) x) * 3;
	                *ptr1 = 255;
	                ptr1 = image.ptr((int) ((0.1) * 1000)) + ((int) x) * 3;
					*ptr1 = 255;
//					if(input[0] == t_x[0] || input[0] == t_x[1] || input[0] == t_x[2])
//					{
//						std::cout << input[0] << " " << y << "\n";
//					}
				}
	            imshow("Network", image);
	            if (missSum / x.size() < 0.05 ) break;
	            cv::waitKey(20);
            }

        }
        cv::waitKey(5000);
        return iteration;
    }

    /*
	 *
	 */
	void testSimpleSigmoid1LayerCPU()
	{
		NeuralNetworkCPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkCPU::InputLayer(1));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(13.2,0.01, 1, nn.getLastLayerNeuronRef()));

		long iteration = testNeuralNetwork({0.6, 0.8},
										   {0.4, 0.9}, &nn);
		std::cout << "Done: " << iteration << "\n";
	}

	/*
	 *
	 */
	void testSimpleSigmoid2LayersCPU()
	{
		NeuralNetworkCPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkCPU::InputLayer(1));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(5.2,0.003, 15, nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(5.2,0.01, 1, nn.getLastLayerNeuronRef()));

		long iteration = testNeuralNetwork({0.6, 0.8},
										   {0.4, 0.9}, &nn);
		std::cout << "Done: " << iteration << "\n";
	}

	/*
	 *
	 */
	void testPeak2LayersCPU()
	{
		NeuralNetworkCPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkCPU::InputLayer(1));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(16.2,0.008, 10, nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(16.2,0.012, 1, nn.getLastLayerNeuronRef()));

		long iteration = testNeuralNetwork({0.6, 0.7, 0.8},
										   {0.4, 0.9, 0.4}, &nn);
		std::cout << "Done: " << iteration << "\n";
	}

	/*
	 *
	 */
	void testPeak3LayersCPU()
	{
		NeuralNetworkCPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkCPU::InputLayer(1));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(16.2,0.006, 30, nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(16.2,0.008, 20, nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(16.2,0.012, 1, nn.getLastLayerNeuronRef()));

		long iteration = testNeuralNetwork({0.6, 0.7, 0.8},
										   {0.4, 0.9, 0.4}, &nn);
		std::cout << "Done: " << iteration << "\n";
	}

	void testNNSpeedCPU()
	{
		std::vector<double> x = {1};
		std::vector<double> z = {0.5};
		NeuralNetworkCPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkCPU::InputLayer(1));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(13.2,20.2,1000,nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(13.2,30.2,1000, nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkCPU::SigmoidLayer(13.2,30.2, 1, nn.getLastLayerNeuronRef()));
		int64 timeBefore = cv::getTickCount();
		long i;
		for (i = 0; i < 1000; i++)
		{
			nn.determineOutput(x);
			nn.learnBackPropagation(z);
			nn.determineOutput(z);
		}

		int64 timeAfter = cv::getTickCount();
		std::cout << (timeAfter - timeBefore) / cv::getTickFrequency() << "\n";
	}
}

#endif //QBOT_NNTEST_H
