/*
 * NeuralNetwork.cpp
 *
 *  Created on: 9 maj 2020
 *      Author: przemo
 */

#include "NeuralNetwork.h"

namespace NeuralNetworkGPU {

	/*
	 *
	 */
	NeuralNetwork::NeuralNetwork(LearnMode t_lm)
	{
		learnMode = t_lm;
		srand (time(NULL));
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
	NeuronsPtr NeuralNetwork::getLastLayerNeuronRef()
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
	std::vector<double> NeuralNetwork::determineOutput(std::vector<double> x)
	{
		//Prepare input
		int lastValueIndex = 0;
		std::list<NNLayer*>::iterator layer = layers.begin();
		while(x.size() > lastValueIndex && layer != layers.end())
		{
//			std::cout << x.size() << "   "  << lastValueIndex << "   "  << (*layer)->getNeuronPtr().size << "\n";
			assert(x.size() >= lastValueIndex+(*layer)->getNeuronPtr().size && "Input size is not okie dokie (double)");
			std::vector<double>::const_iterator first = x.begin() + lastValueIndex;
			std::vector<double>::const_iterator last = x.begin() + lastValueIndex+(*layer)->getNeuronPtr().size;
			std::vector<double> input(first,last);
			(*layer)->setInput(input);

			lastValueIndex += (*layer)->getNeuronPtr().size;
			layer++;
		}

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
	std::vector<double> NeuralNetwork::determineOutput(std::vector<int> x)
	{
		//Prepare input
		int lastValueIndex = 0;
		std::list<NNLayer*>::iterator layer = layers.begin();
		while(x.size() > lastValueIndex && layer != layers.end())
		{
			assert(x.size() >= lastValueIndex+(*layer)->getNeuronPtr().size && "Input size is not okie dokie (int)");
			std::vector<int>::const_iterator first = x.begin() + lastValueIndex;
			std::vector<int>::const_iterator last = x.begin() + lastValueIndex+(*layer)->getNeuronPtr().size;
			std::vector<int> input(first,last);
			(*layer)->setInput(input);

			lastValueIndex += (*layer)->getNeuronPtr().size;
			layer++;
		}

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

		if(learnMode == LearnMode::SGD)
		{
			for(auto it=layers.rbegin(); it!=layers.rend(); it++)
			{
				(*it)->learnSGD();
			}
		}
		else if(learnMode == LearnMode::Adam)
		{
			for(auto it=layers.rbegin(); it!=layers.rend(); it++)
			{
				(*it)->learnAdam();
			}
		}
	}

	/*
	 *
	 */
	void NeuralNetwork::saveToFile()
	{
//		std::remove("NeuralNetwork.dat");
//		std::ofstream file("NeuralNetwork.dat");
//
//		file << layers.size() << ' ';
//		for(auto it : layers)
//		{
//			it->saveToFile(file);
//		}
//
//		file.close();
	}

	/*
	 *
	 */
	void NeuralNetwork::loadFromFile()
	{
//		std::ifstream file("NeuralNetwork.dat");
//
//		for(auto it = layers.begin(); it != layers.end(); it++)
//		{
//			delete (*it);
//		}
//		layers.clear();
//
//		int numberOfLayers,layerId;
//		file >> numberOfLayers;
//
//		while(file >> layerId)
//		{
//			if(layerId == 0) // InputLayer
//			{
//				int size;
//				file >> size;
//				addLayer(new InputLayer(size));
//			}
//			else if (layerId == 1) //Sigmoid Layer
//			{
//				int size;
//				double learnRate, b;
//				file >> size;
//				file >> learnRate;
//				file >> b;
//
//				SigmoidLayer::configure(b);
//				addLayer(new SigmoidLayer(learnRate, size, getLastLayerNeuronRef()));
//			}
//		}
//
//		file.close();
	}

} /* namespace NeuralNetworkGPU */
