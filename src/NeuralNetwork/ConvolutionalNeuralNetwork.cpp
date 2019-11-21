/*
 * ConvolutionalNeuralNetwork.cpp
 *
 *  Created on: 11 lis 2019
 *      Author: przemo
 */

#include "ConvolutionalNeuralNetwork.h"

/*
 *
 */
ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(LayerInfo t_inputSize, std::vector<LayerInfo> t_layers,
													   std::vector<int> filterSize, std::vector<double> t_n, double t_b)
{
	n = t_n;
	b = t_b;

	inputLayer.clear();
	for(int i=0; i<t_inputSize.depth*t_inputSize.height*t_inputSize.width; i++)
	{
		InputNeuron inputNeuron = InputNeuron();
		inputLayer.push_back(inputNeuron);
	}

	int heightDiff = (t_inputSize.height-t_layers[0].height)/2;
	int widthDiff = (t_inputSize.width-t_layers[0].width)/2;
	std::vector<Neuron> layer;
	for(int z=0; z<t_layers[0].depth; z++)
	{
		for(int y=0; y<t_layers[0].height; y++)
		{
			for(int x=0; x<t_layers[0].width; x++)
			{
				//Creating first hidden layer
				std::vector<Neuron*> prevLayerReference;
				for(int zz=0; zz<t_inputSize.depth; zz++)
				{
					for(int yy=-filterSize[0]; yy<=filterSize[0]; yy++)
					{
						for(int xx=-filterSize[0]; xx<=filterSize[0]; xx++)
						{
							if(heightDiff+y+yy>=0 && heightDiff+y+yy<t_inputSize.height &&
							    widthDiff+x+xx>=0 && widthDiff+x+xx<t_inputSize.width)
							{
								signed long index = (zz)*t_inputSize.width*t_inputSize.height +
													(heightDiff+y+yy)*t_inputSize.width +
													(widthDiff+x+xx);
								prevLayerReference.push_back(&(inputLayer[index]));
							}
						}
					}
				}
				Neuron k = Neuron(prevLayerReference, &(n[0]), &b);
				layer.push_back(k);
			}
		}
	}
	hiddenLayers.push_back(layer);

	//Creating other layers
	for(int i=1; i<t_layers.size(); i++)
	{
		std::vector<Neuron> layer;
		int heightDiff = (t_layers[i-1].height-t_layers[i].height)/2;
		int widthDiff = (t_layers[i-1].width-t_layers[i].width)/2;
		for(int z=0; z<t_layers[i].depth; z++)
		{
			for(int y=0; y<t_layers[i].height; y++)
			{
				for(int x=0; x<t_layers[i].width; x++)
				{
					//Creating first hidden layer
					std::vector<Neuron*> prevLayerReference;
					for(int zz=0; zz<t_layers[i-1].depth; zz++)
					{
						for(int yy=-filterSize[i]; yy<=filterSize[i]; yy++)
						{
							for(int xx=-filterSize[i]; xx<=filterSize[i]; xx++)
							{
								if(heightDiff+y+yy>=0 && heightDiff+y+yy<t_layers[i-1].height &&
								    widthDiff+x+xx>=0 && widthDiff+x+xx<t_layers[i-1].width)
								{
									signed long index = (zz)*t_layers[i-1].width*t_layers[i-1].height +
														(heightDiff+y+yy)*t_layers[i-1].width +
														(widthDiff+x+xx);
									prevLayerReference.push_back(&(hiddenLayers.back()[index]));
								}
							}
						}
					}
					Neuron k = Neuron(prevLayerReference, &(n[i]), &b);
					layer.push_back(k);
				}
			}
		}
		hiddenLayers.push_back(layer);
	}

#ifdef PRINT_CONV_NEURON_CONNETIONS
	std::cout << "Input\n";
	for(std::vector<InputNeuron>::iterator it=inputLayer.begin(); it!=inputLayer.end(); it++)
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

ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork()
{

}

