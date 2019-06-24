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
NeuralNetwork::NeuralNetwork(int t_inputSize, std::vector<int> t_layers,std::vector<double> t_n, double t_b)
{
	n = t_n;
	b = t_b;

	inputLayer.clear();
	InputNeuron inputNeuron = InputNeuron(1.0);
	inputLayer.push_back(inputNeuron);
	for(int i=0; i<t_inputSize; i++)
	{
		inputNeuron = InputNeuron();
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
		prevLayerReference.push_back(&(*inputLayer.begin())); //Bias in other layers
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
	for(std::list<std::vector<Neuron>>::iterator iterator=hiddenLayers.begin(); iterator!=hiddenLayers.end(); iterator++)
	{
		std::cout << "Layer: " << j << "\n";j++;
		for(std::vector<Neuron>::iterator it_neuron=iterator->begin(); it_neuron!=iterator->end(); it_neuron++)
		{
			std::cout << &(*it_neuron) << ":      ";
			for(int i=0; i<it_neuron->input.size(); i++) std::cout << it_neuron->input[i] << "  ";
			std::cout << "\n";
		}
	}
#endif
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& t_nn) : NeuralNetwork(t_nn.getInputSize(), t_nn.getLayersLayout(),t_nn.getLearningRates(),t_nn.getActivationFunctionParameter())
{
	const std::list<std::vector<Neuron>>* srcHiddenLayers = t_nn.getHiddenLayers();

	std::list<std::vector<Neuron>>::iterator it_layer=hiddenLayers.begin();
	std::list<std::vector<Neuron>>::const_iterator it_srcLayer=srcHiddenLayers->begin();
	while(it_layer!=hiddenLayers.end())
	{
		for(int i=0; i<it_layer->size(); i++)
		{
			(*it_layer)[i].setW((*it_srcLayer)[i].getW());
		}
		it_srcLayer++;
		it_layer++;
	}
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
std::vector<double> NeuralNetwork::determineY(std::vector<double> &x)
{
	//Prepare input
	if(x.size() != inputLayer.size()-1)
	{
		std::cout << x.size() << "  " << inputLayer.size()-1 << "\n";
		assert(x.size() == inputLayer.size()-1);
	}
	int i = 0;
	for(std::list<InputNeuron>::iterator it_input=++inputLayer.begin(); it_input!=inputLayer.end(); it_input++)
	{
		it_input->setY(x[i]);
		i++;
	}

	return determineY();
}

/*
 *
 */
std::vector<double> NeuralNetwork::determineY(const std::vector<int> &x)
{
	//Prepare input
	if(x.size() != inputLayer.size()-1)
	{
		std::cout << x.size() << "  " << inputLayer.size()-1 << "\n";
		assert(x.size() == inputLayer.size()-1);
	}
	int i = 0;
	for(std::list<InputNeuron>::iterator it_input=++inputLayer.begin(); it_input!=inputLayer.end(); it_input++)
	{
		it_input->setY(x[i]);
		i++;
	}

	return determineY();
}

/*
 *
 */
std::vector<double> NeuralNetwork::determineY()
{
	//Calculate
	for(std::list<std::vector<Neuron>>::iterator it_layer=hiddenLayers.begin(); it_layer!=hiddenLayers.end(); it_layer++)
	{
		int i;
		#pragma omp parallel for shared(it_layer) private(i)
		for(i=0; i<it_layer->size(); i++)
		{
			(*it_layer)[i].determineY();
		}

	}

	//Prepare result
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
void NeuralNetwork::learnBackPropagation(std::vector<double> &z)
{
	assert(z.size() == hiddenLayers.back().size());
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
			(*it_layer)[i].calculateDerative();
		}
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

/*
 *
 */
void NeuralNetwork::learnBackPropagation(std::vector<double> &x, std::vector<double> &z)
{
	determineY(x);
	learnBackPropagation(z);
}

/*
 *
 */
std::vector<int> NeuralNetwork::getLayersLayout() const
{
	std::vector<int> result;
	for(std::list<std::vector<Neuron>>::const_iterator it_layer=hiddenLayers.begin(); it_layer!=hiddenLayers.end(); it_layer++)
	{
		result.push_back(it_layer->size());
	}
	return result;
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
		if(fabs(inputNeuron.getY()>0.5)) t = -1;
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
			if(fabs(neuron.getY()>0.5)) t = -1;
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
