/*
 * SiecNeuronowa.cpp
 *
 *  Created on: 20 gru 2017
 *      Author: przemo
 */

#include "NeuralNetwork.h"

/*
 *
 */
NeuralNetwork::NeuralNetwork()
{

}

/*
 *
 */
NeuralNetwork::~NeuralNetwork()
{
    for(auto it = layers.begin(); it != layers.end(); it++)
    {
        if((*it) != nullptr) delete (*it);
    }
}

/*
 *
 */
void NeuralNetwork::addLayer(NNLayer *t_newLayer)
{
    layers.push_back(t_newLayer);
}

/*
 *
 */
std::vector<Neuron*> NeuralNetwork::getLastLayerNeuronRef()
{
    return (*layers.rbegin())->getNeuronPtr();
}

/*
 *
 */
TensorSize NeuralNetwork::getLastLayerTensorSize()
{
	return (*layers.rbegin())->getTensorOutputSize();
}

/*
 *
 */
std::vector<double> NeuralNetwork::determineOutput(std::vector<double> &x)
{
    //Prepare input
    (*layers.begin())->setInput(x);

    //Calculate
    for(auto it = layers.begin(); it != layers.end(); it++)
    {
        (*it)->determineOutput();
    }

    return (*layers.rbegin())->getOutput();
}

/*
 *
 */
std::vector<double> NeuralNetwork::determineOutput(const std::vector<int> &x)
{
	//Prepare input
    (*layers.begin())->setInput(x);

    //Calculate
    for(auto it = layers.begin(); it != layers.end(); it++)
    {
        (*it)->determineOutput();
    }

    return (*layers.rbegin())->getOutput();
}

/*
 *
 */
std::vector<double> NeuralNetwork::getOutput()
{
    return (*layers.rbegin())->getOutput();
}

/*
 *
 */
void NeuralNetwork::learnBackPropagation(std::vector<double> &z)
{
    (*layers.rbegin())->setDelta(z);

    for(auto it=layers.rbegin(); it!=layers.rend(); it++)
    {
        (*it)->learnBackPropagation();
    }
}

/*####################################################################################*/

/*
 *
 */
//void NeuralNetwork::displayNeuralNetwork()
//{
//	cv::Mat image;
//	image = cv::Mat(1000, 800, CV_8UC3);
//
//	for(int y = 0 ; y < image.rows ; y++)
//	{
//		uchar* ptr = image.ptr((int)y);
//		for(int x = 0 ; x < image.cols*3 ; x++)
//		{
//			*ptr=0;
//			ptr = ptr+1;
//		}
//	}
//
//	//Input Layer
//	int y = 0;
//	for(InputNeuron inputNeuron : inputLayer)
//	{
//		cv::Point p = cv::Point(15,15+15*y);
//		int c = ((double) inputNeuron.getOutput()) * 255;
//		int t = 1;
//		if(fabs(inputNeuron.getOutput() > 0.5)) t = -1;
//		if(c>=0) cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(255, c, 0), t);
//		else cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(255, 0, -c), t);
//		y++;
//	}
//
//	//Other Layers
//	int x = 0;
//	for(std::vector<AdaptiveNeuron> layer : hiddenLayers)
//	{
//		int y = 0;
//		for(AdaptiveNeuron neuron : layer)
//		{
//			cv::Point p = cv::Point(30+15*x,15+15*y);
//			int c = ((double) neuron.getOutput()) * 255;
//			int t = 1;
//			if(fabs(neuron.getOutput() > 0.5)) t = -1;
//			if(c>=0) cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(0, c, 0), t);
//			else cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(0, 0, -c), t);
//			y++;
//		}
//		x++;
//	}
//
//	imshow("Network", image);
//	cv::waitKey(20);
//}
