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
NeuralNetwork::NeuralNetwork(std::vector<double*> t_input, std::vector<int> t_layers,std::vector<double> t_n, double t_b)
{
	oneValue = 1.0;

	n = t_n;
	b = t_b;

	inputLayer.clear();
	InputNeuron inputNeuron = InputNeuron(&oneValue);
	inputLayer.push_back(inputNeuron);
	for(int i=0; i<t_input.size(); i++)
	{
		inputNeuron = InputNeuron(t_input[i]);
		inputLayer.push_back(inputNeuron);
	}

	//Creating first hidden layer
	std::vector<Neuron*> prevLayerReference;
	for(std::list<InputNeuron>::iterator it=inputLayer.begin(); it!=inputLayer.end(); it++)
	{
		prevLayerReference.push_back(&(*it));
	}

	std::vector<Neuron> layer;
	for(int i=0; i<t_layers[0]; i++)
	{
		Neuron k = Neuron(prevLayerReference, &(n[0]), &b);
		layer.push_back(k);
	}
	hiddenLayers.push_back(layer);

	//Creating other layers
	for(int i=1; i<t_layers.size(); i++)
	{
		//Connections
		std::vector<Neuron*> prevLayerReference;
		prevLayerReference.push_back(&(*inputLayer.begin()));
		for(std::vector<Neuron>::iterator it=hiddenLayers.back().begin(); it!=hiddenLayers.back().end(); it++)
		{
			prevLayerReference.push_back(&(*it));
		}
		//Neurons
		std::vector<Neuron> layer;
		layer.clear();
		for(int j=0; j<t_layers[i]; j++)
		{
			Neuron neuron = Neuron(prevLayerReference,&(n[i]),&b);
			layer.push_back(neuron);
		}
		hiddenLayers.push_back(layer);
	}

#ifdef PRINT_NEURON_CONNETIONS
	std::cout << "Input\n";
	for(std::list<InputNeuron>::iterator it=inputLayer.begin(); it!=inputLayer.end(); it++)
	{
		std::cout << &(*it) << "   ";
	}
	std::cout << "\n";
	std::cout << "Core\n";
	int j = 0;
	for(std::list<std::list<Neuron>>::iterator iterator=hiddenLayers.begin(); iterator!=hiddenLayers.end(); iterator++)
	{
		std::cout << "Layer: " << j << "\n";j++;
		for(std::list<Neuron>::iterator it_neuron=iterator->begin(); it_neuron!=iterator->end(); it_neuron++)
		{
			std::cout << &(*it_neuron) << ":      ";
			for(int i=0; i<it_neuron->input.size(); i++) std::cout << it_neuron->input[i] << "  ";
			std::cout << "\n";
		}
	}
#endif
}

/*
 *
 */
NeuralNetwork::~NeuralNetwork()
{

}

/*
 *
 */
std::vector<double> NeuralNetwork::determineY()
{
	for(std::list<std::vector<Neuron>>::iterator it_layer=hiddenLayers.begin(); it_layer!=hiddenLayers.end(); it_layer++)
	{
		int i;
		#pragma omp parallel for shared(it_layer) private(i)
		for(i=0; i<it_layer->size(); i++)
		{
			(*it_layer)[i].determineY();
		}

	}

	std::vector<double> result;
	for(std::vector<Neuron>::iterator it_neuron=hiddenLayers.back().begin(); it_neuron!=hiddenLayers.back().end(); it_neuron++)
	{
		result.push_back(it_neuron->getY());
	}
	return result;
}

/*
 *
 */
std::vector<double> NeuralNetwork::getY()
{
	std::vector<double> wynik;
	for(std::vector<Neuron>::iterator it_neuron=hiddenLayers.back().begin(); it_neuron!=hiddenLayers.back().end(); it_neuron++)
	{
		wynik.push_back(it_neuron->getY());
	}

	return wynik;
}

/*
 *
 */
void NeuralNetwork::learnBackPropagation(std::vector<double> z)
{
//	modifyLearningRate();

	assert("z.size() != hiddenLayer.back().size())" && z.size() == hiddenLayers.back().size());
	int i=0;
	for(std::vector<Neuron>::iterator it=hiddenLayers.back().begin(); it!=hiddenLayers.back().end(); it++,i++)
	{
		it->setDelta(z[i] - it->getY());
	}

	for(std::list<std::vector<Neuron>>::reverse_iterator it_layer=hiddenLayers.rbegin(); it_layer!=hiddenLayers.rend(); it_layer++)
	{
		#pragma omp parallel for shared(it_layer) private(i)
		for(i=0; i<it_layer->size(); i++)
		{
			(*it_layer)[i].learnDeltaRule();
		}
	}
}

void NeuralNetwork::modifyLearningRate(double v)
{
	n[0] = v*0.00001;
	n[1] = v*0.00003;
	n[2] = v*0.0001;
	n[3] = v*0.0003;
	n[4] = v*0.001;
	n[5] = v*0.003;
	n[6] = v*0.01;
	n[7] = v*0.03;
	n[8] = v*0.1;
}

/*####################################################################################*/

/*
 *
 */
void NeuralNetwork::displayNeuralNetwork()
{
	cv::Mat image;
	image = cv::Mat(1000, 800, CV_8UC3);

	for(int y = 0 ; y < image.rows ; y++)
	{
		uchar* ptr = image.ptr((int)y);
		for(int x = 0 ; x < image.cols*3 ; x++)
		{
			*ptr=0;
			ptr = ptr+1;
		}
	}

	//Input Layer
	int y = 0;
	for(InputNeuron inputNeuron : inputLayer)
	{
		cv::Point p = cv::Point(15,15+15*y);
		int c = ((double)inputNeuron.getY())*255;
		int t = 1;
		if(fabs(inputNeuron.getY()>0.5)) t = CV_FILLED;
		if(c>=0) cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(255, c, 0), t);
		else cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(255, 0, -c), t);
		y++;
	}

	//Other Layers
	int x = 0;
	for(std::vector<Neuron> layer : hiddenLayers)
	{
		int y = 0;
		for(Neuron neuron : layer)
		{
			cv::Point p = cv::Point(30+15*x,15+15*y);
			int c = ((double)neuron.getY())*255;
			int t = 1;
			if(fabs(neuron.getY()>0.5)) t = CV_FILLED;
			if(c>=0) cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(0, c, 0), t);
			else cv::rectangle(image, p, p+cv::Point(9,9), cv::Scalar(0, 0, -c), t);
			y++;
		}
		x++;
	}

	imshow("Network", image);
	cv::waitKey(20);
}

/*
 *
 */
void NeuralNetwork::writeNeuronsToFile()
{
	std::ofstream file;
	file.open("Neuron.txt");

	int layerNumber = 0;
	for(std::vector<Neuron> it_layer : hiddenLayers)
	{
		layerNumber++;
		file << "Layer no." << layerNumber << "\n";
		for(Neuron neuron : it_layer)
		{
			std::vector<double> w = neuron.getW();
			for(double v : w)
			{
				file << v << "  ";
			}
			file << "\n";
		}
		file << "\n";
	}
	file.close();
}

/*
 *
 */
void NeuralNetwork::printNeuralNetworkInfo()
{
	for(int i=0; i<n.size(); i++)
	{
		std::cout << n[i] << "  ";
	}
	std::cout << "\n";
}
