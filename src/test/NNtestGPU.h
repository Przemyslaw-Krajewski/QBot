//
// Created by przemo on 02.01.2020.
//

#ifndef QBOT_NNTESTGPU_H
#define QBOT_NNTESTGPU_H

#include "../NeuralNetworkGPU/NeuralNetwork.h"
#include "../NeuralNetwork/NeuralNetwork.h"

namespace Test
{

	void gpuSigmoidNNSmokeTest()
	{
		NeuralNetworkGPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkGPU::InputLayer(3));
		nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,0.3,1, nn.getLastLayerNeuronRef()));

		std::vector<double> in;
		in.push_back(0.4);
		in.push_back(7.7);
		in.push_back(3.2);

		std::cout << "Input:\n";
		for(double i : in) std::cout << "   " << i << "\n";

		nn.determineOutput(in);
		std::vector<double> out = nn.getOutput();
		std::cout << "Output:\n";
		for(double i : out) std::cout << "   " << i << "\n";

		std::vector<double> z = {0.5};
        nn.setMeanSquareDelta(z);
        nn.learnBackPropagation();

		out = nn.determineOutput(in);
		std::cout << "Output:\n";
		for(double i : out) std::cout << "   " << i << "\n";
	}

	void gpuConvNNSmokeTest()
	{
		NeuralNetworkGPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(3,3,1)));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.001,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));

		std::vector<double> in;
		in.push_back(1);in.push_back(2);in.push_back(3);
		in.push_back(4);in.push_back(5);in.push_back(6);
		in.push_back(7);in.push_back(8);in.push_back(9);

		std::cout << "Input:\n";
		for(double i : in) std::cout << "   " << i << "\n";

		nn.determineOutput(in);
		std::vector<double> out = nn.getOutput();
		std::cout << "Output:\n";
		for(double i : out) std::cout << "   " << i << "\n";

		std::vector<double> z = {300};
		nn.setMeanSquareDelta(z);
        nn.learnBackPropagation();

		out = nn.determineOutput(in);
		std::cout << "Output:\n";
		for(double i : out) std::cout << "   " << i << "\n";
	}

    /*
     *
     */
    long testNeuralNetwork(std::vector<std::vector<double>> t_x, std::vector<double> t_z, NeuralNetworkGPU::NeuralNetwork *nn)
    {
        std::vector<std::vector<double>> x;
        for (std::vector<double> &i : t_x) x.push_back(i);
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
                nn->determineOutput(x[i]);
                nn->setMeanSquareDelta(z[i]);
                nn->learnBackPropagation();
                double value = nn->determineOutput(x[i])[0];
                double miss = value - z[i][0];
                missSum += fabs(miss);
                std::cout << "Miss: "<< miss << "\n";
            }

            if(x[0].size() == 1 && (iteration %100 == 0 || missSum / x.size() < 0.01))
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
				}

				imshow("Network", image);
				cv::waitKey(20);
			}
            if (missSum / x.size() < 0.01 || iteration > 15000) break;
        }
        cv::waitKey(4000);
        return iteration;
    }

    /*
     *
     */
    void testSimpleSigmoid1LayerGPU()
    {
        NeuralNetworkGPU::NeuralNetwork nn;
        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,0.01, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.6}, {0.8}},
                                           {0.4, 0.9}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    /*
     *
     */
    void testSimpleSigmoid2LayersGPU()
    {
    	NeuralNetworkGPU::NeuralNetwork nn;
        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(5.2,0.003, 15, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(5.2,0.01, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.6}, {0.8}},
                                           {0.4, 0.9}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    /*
     *
     */
    void testPeak2LayersGPU()
    {
    	NeuralNetworkGPU::NeuralNetwork nn;
        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(16.2,0.008, 10, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(16.2,0.012, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.6}, {0.7}, {0.8}},
                                           {0.4, 0.9, 0.4}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    /*
     *
     */
    void testPeak3LayersGPU()
    {
    	NeuralNetworkGPU::NeuralNetwork nn;
        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(16.2,0.006, 30, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(16.2,0.008, 20, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(16.2,0.012, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.6}, {0.7}, {0.8}},
                                           {0.4, 0.9, 0.4}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    /*
     *
     */
    void testLargeLayerGPU()
    {
    	NeuralNetworkGPU::NeuralNetwork nn;
        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(16.2,0.006, 3800, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(0.01,0.008, 10, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(16.2,0.012, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.6}, {0.7}, {0.8}},
                                           {0.4, 0.9, 0.4}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    void testNNSpeedGPU()
    {
        std::vector<double> x = {1};
        std::vector<double> z = {0.5};
        NeuralNetworkGPU::NeuralNetwork nn;
        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,20.2,4000, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,30.2,4000, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,30.2, 1, nn.getLastLayerNeuronRef()));
        int64 timeBefore = cv::getTickCount();
        long i;
        for (i = 0; i < 1000; i++)
        {
            nn.determineOutput(x);
            nn.setMeanSquareDelta(z);
            nn.learnBackPropagation();
            nn.determineOutput(z);
        }

        int64 timeAfter = cv::getTickCount();
        std::cout << (timeAfter - timeBefore) / cv::getTickFrequency() << "\n";
    }

    void testNNSpeedAdamGPU()
    {
        std::vector<double> x = {1};
        std::vector<double> z = {0.5};
        NeuralNetworkGPU::NeuralNetwork nn(NeuralNetworkGPU::LearnMode::Adam);
        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,20.2,4000, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,30.2,4000, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,30.2, 1, nn.getLastLayerNeuronRef()));
        int64 timeBefore = cv::getTickCount();
        long i;
        for (i = 0; i < 1000; i++)
        {
            nn.determineOutput(x);
            nn.setMeanSquareDelta(z);
            nn.learnBackPropagation();
            nn.determineOutput(z);
        }

        int64 timeAfter = cv::getTickCount();
        std::cout << (timeAfter - timeBefore) / cv::getTickFrequency() << "\n";
    }

    /*
     *
     */
    void testPeak3LayersAdamGPU()
    {
    	NeuralNetworkGPU::NeuralNetwork nn(NeuralNetworkGPU::LearnMode::Adam);

        nn.addLayer(new NeuralNetworkGPU::InputLayer(1));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(11.2,0.0006, 30, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(9.2,0.0008, 20, nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(5.2,0.0012, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.6}, {0.7}, {0.8}},
                                           {0.4, 0.9, 0.4}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

	void testConv1ValueGPU()
	{
		NeuralNetworkGPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(3,3,1)));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.01,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3}},
                                           {40}, &nn);
        std::cout << "Done: " << iteration << "\n";
	}

	void testConv2ValuesGPU()
	{
		NeuralNetworkGPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(3,3,1)));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.01,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3},
        									{0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1}},
                                           {40,100}, &nn);
        std::cout << "Done: " << iteration << "\n";
	}

	void testConv2ConvValuesGPU()
	{
		NeuralNetworkGPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(3,3,2)));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.01,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3,0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3},
        									{0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1,0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1}},
                                           {40,100}, &nn);
        std::cout << "Done: " << iteration << "\n";
	}

	void testConv2LayersGPU()
	{
		NeuralNetworkGPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(5,5,1)));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.004,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.001,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5},
        									{0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1}},
                                           {40,100}, &nn);
        std::cout << "Done: " << iteration << "\n";
	}

	void testConvAndSigmoidGPU()
	{
		NeuralNetworkGPU::NeuralNetwork nn;
		nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(3,3,1)));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.01,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(11.2,0.03, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3},
										    {0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1},
											{0.8, 0.2, 0.1, 0.8, 0.2, 0.1, 0.8, 0.2, 0.1}},
                                           {0.8, 0.1, 0.8}, &nn);
        std::cout << "Done: " << iteration << "\n";
	}

	void testFuseLayersGPUSmokeTest()
	{
        NeuralNetworkGPU::NeuralNetwork nn;
        NeuralNetworkGPU::InputLayer* inputLayer1 = new NeuralNetworkGPU::InputLayer(1);
        NeuralNetworkGPU::InputLayer* inputLayer2 = new NeuralNetworkGPU::InputLayer(1);
        nn.addLayer(inputLayer1);
        nn.addLayer(inputLayer2);
        nn.addLayer(new NeuralNetworkGPU::FuseLayer(inputLayer1->getNeuronPtr(),inputLayer2->getNeuronPtr()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,0.01, 1, nn.getLastLayerNeuronRef()));

        std::vector<double> in;
		in.push_back(0);in.push_back(2);

		std::cout << "Input:\n";
		for(double i : in) std::cout << "   " << i << "\n";

		nn.determineOutput(in);
		std::vector<double> out = nn.getOutput();
		std::cout << "Output:\n";
		for(double i : out) std::cout << "   " << i << "\n";

		std::vector<double> z = {0.5};
        nn.setMeanSquareDelta(z);
        nn.learnBackPropagation();

		out = nn.determineOutput(in);
		std::cout << "Output:\n";
		for(double i : out) std::cout << "   " << i << "\n";
	}

	void testFuseLayersGPU()
	{
        NeuralNetworkGPU::NeuralNetwork nn(NeuralNetworkGPU::LearnMode::Adam);
        NeuralNetworkGPU::InputLayer* inputLayer1 = new NeuralNetworkGPU::InputLayer(1);
        NeuralNetworkGPU::InputLayer* inputLayer2 = new NeuralNetworkGPU::InputLayer(1);
        nn.addLayer(inputLayer1);
        nn.addLayer(inputLayer2);
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(11.2,0.0006, 30, inputLayer1->getNeuronPtr()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(9.2,0.0008, 20, nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkGPU::FuseLayer(nn.getLastLayerNeuronRef(),inputLayer2->getNeuronPtr()));
        nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(5.2,0.0012, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0,1.6}, {0,2.8}},
                                           {0.4, 0.9}, &nn);
	}

    void testConvNNSpeedGPU()
    {
        std::vector<double> x = std::vector<double>(20*20);
        std::vector<double> z = {0.5};
        NeuralNetworkGPU::NeuralNetwork nn(NeuralNetworkGPU::LearnMode::Adam);
        nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(20,20,1)));
        nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.01,100,NeuralNetworkGPU::MatrixSize(5,5),nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.01,100,NeuralNetworkGPU::MatrixSize(5,5),nn.getLastLayerNeuronRef()));
        nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,0.01,10,NeuralNetworkGPU::MatrixSize(5,5),nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>(13.2,30.2, 1, nn.getLastLayerNeuronRef()));
        int64 timeBefore = cv::getTickCount();
        long i;
        for (i = 0; i < 100; i++)
        {
            nn.determineOutput(x);
            nn.setMeanSquareDelta(z);
            nn.learnBackPropagation();
            nn.determineOutput(x);
        }

        int64 timeAfter = cv::getTickCount();
        std::cout << (timeAfter - timeBefore) / cv::getTickFrequency() << "\n";
    }

	void testConv2LayersAdamGPU()
	{
		NeuralNetworkGPU::NeuralNetwork nn(NeuralNetworkGPU::LearnMode::Adam);
		nn.addLayer(new NeuralNetworkGPU::InputLayer(NeuralNetworkGPU::TensorSize(5,5,1)));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,4.4,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));
		nn.addLayer(new NeuralNetworkGPU::ConvolutionalLayer(0.0,1.1,1,NeuralNetworkGPU::MatrixSize(3,3),nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({{0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5},
        									{0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1}},
                                           {40,100}, &nn);
        std::cout << "Done: " << iteration << "\n";
	}

}

#endif //QBOT_NNTESTGPU_H
