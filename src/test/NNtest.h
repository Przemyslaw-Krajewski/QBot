//
// Created by przemo on 02.01.2020.
//

#ifndef QBOT_NNTEST_H
#define QBOT_NNTEST_H

#include "../NeuralNetwork/NeuralNetwork.h"
#include <cmath>

namespace Test
{

    /*
     *
     */
    long testNeuralNetwork(std::vector<double> t_x, std::vector<double> t_z, NeuralNetwork *nn)
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
                nn->determineOutput(x[i]);
                nn->learnBackPropagation(z[i]);
                double miss = nn->determineOutput(x[i])[0] - z[i][0];
                missSum += fabs(miss);
                std::cout << miss << "\n";
            }

            if (missSum / x.size() < 0.05) break;

            for (int y = 0; y < image.rows; y++)
            {
                uchar *ptr = image.ptr((int) y);
                for (int x = 0; x < image.cols * 3; x++)
                {
                    *ptr = 0;
                    ptr = ptr + 1;
                }
            }

            for (int x = 0; x < 1000; x += 1)
            {
                std::vector<double> input = std::vector<double>{((double) x) / 1000};
                double y = nn->determineOutput(input)[0];
                uchar *ptr1 = image.ptr((int) ((1 - y) * 999)) + ((int) x) * 3;
                uchar *ptr2 = image.ptr((int) ((1 - y) * 999)) + ((int) x) * 3 + 1;
                uchar *ptr3 = image.ptr((int) ((1 - y) * 999)) + ((int) x) * 3 + 2;

                *ptr1 = 255;
                *ptr2 = 255;
                *ptr3 = 255;

                ptr1 = image.ptr((int) ((0.4) * 1000)) + ((int) x) * 3;
                *ptr1 = 255;
                ptr1 = image.ptr((int) ((0.9) * 1000)) + ((int) x) * 3;
                *ptr1 = 255;
            }

            imshow("Network", image);
            cv::waitKey(20);
        }
        return iteration;
    }

    /*
     *
     */
    long testNeuralNetworkConsole(std::vector<std::vector<double>> t_x, std::vector<double> t_z, NeuralNetwork *nn)
    {
        std::vector<std::vector<double>> x;
        for (std::vector<double> &i : t_x) x.push_back(i);
        std::vector<std::vector<double>> z;
        for (double &i : t_z) z.push_back({i});

        unsigned long long iteration = 0;

        while (1)
        {
            double missSum = 0;
            iteration++;
            std::cout << iteration << "\n";
            for (int i = 0; i < x.size(); i++)
            {
                nn->determineOutput(x[i]);
                nn->learnBackPropagation(z[i]);
                double miss = nn->determineOutput(x[i])[0] - z[i][0];
                missSum += fabs(miss);
                std::cout << miss << "\n";
            }

            if (missSum / x.size() < 0.05) break;
        }
        return iteration;
    }

    /*
     *
     */
    void testSimpleSigmoid1Layer()
    {
        SigmoidLayer::configure(13.2);
        NeuralNetwork nn;
        nn.addLayer(new InputLayer(1));
        nn.addLayer(new SigmoidLayer(0.01, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({0.6, 0.8},
                                           {0.4, 0.9}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    /*
     *
     */
    void testSimpleSigmoid2Layers()
    {
        SigmoidLayer::configure(13.2);
        NeuralNetwork nn;
        nn.addLayer(new InputLayer(1));
        nn.addLayer(new SigmoidLayer(0.003, 15, nn.getLastLayerNeuronRef()));
        nn.addLayer(new SigmoidLayer(0.01, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({0.6, 0.8},
                                           {0.4, 0.9}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    /*
     *
     */
    void testPeak2Layers()
    {
        SigmoidLayer::configure(13.2);
        NeuralNetwork nn;
        nn.addLayer(new InputLayer(1));
        nn.addLayer(new SigmoidLayer(0.003, 15, nn.getLastLayerNeuronRef()));
        nn.addLayer(new SigmoidLayer(0.01, 1, nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetwork({0.6, 0.7, 0.8},
                                           {0.4, 0.9, 0.4}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    /*
     *
     */
    void testSimpleConvLayer()
    {
        SigmoidLayer::configure(13.2);
        NeuralNetwork nn;
        nn.addLayer(new InputLayer(4));
        nn.addLayer(new ConvolutionalLayer(0.033, MatrixSize(2,2),1,TensorSize(2,2,1),nn.getLastLayerNeuronRef()));
        nn.addLayer(new PoolingLayer(MatrixSize(1,1),nn.getLastLayerTensorSize(),nn.getLastLayerNeuronRef()));

        long iteration = testNeuralNetworkConsole({{-1, 1, 1, -1},{1, -1, -1, 1},{1, 1, -1, -1},{-1, -1, 1, 1}},
                                                  {0.1, 0.7,0.3,0.6}, &nn);
        std::cout << "Done: " << iteration << "\n";
    }

    void testNNSpeed()
    {
        std::vector<double> x = {1};
        std::vector<double> z = {0.5};
        SigmoidLayer::configure(13.2);
        NeuralNetwork nn;
        nn.addLayer(new InputLayer(1));
        nn.addLayer(new SigmoidLayer(20000.2,300,nn.getLastLayerNeuronRef()));
        nn.addLayer(new SigmoidLayer(30.2, 600, nn.getLastLayerNeuronRef()));
        nn.addLayer(new SigmoidLayer(30.2, 1, nn.getLastLayerNeuronRef()));
        int64 timeBefore = cv::getTickCount();
        long i;
//        #pragma omp parallel for
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
