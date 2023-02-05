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
		t_newLayer->setLayerId(layers.size());
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
	void NeuralNetwork::setMeanSquareDelta(std::vector<double> &z)
	{
		std::vector<double> delta = getOutput();

		assert(z.size() == delta.size() && "learning values size not match");

		for(int i=0; i<delta.size(); i++) delta[i] = delta[i] - z[i];

		(*layers.rbegin())->setDelta(delta);
	}

	/*
	 *
	 */
	void NeuralNetwork::setSoftMaxDelta(std::vector<double> &z, double diff, int chosen)
	{
		assert(z.size() == getOutput().size() && "learning values size not match");

		std::vector<double> delta = std::vector<double>(z.size(),0);

		//calculate some things
		std::vector<double> s;
		for(int i=0; i<z.size(); i++) s.push_back(exp(z[i]));
		double sSum = 0;
		for(int i=0; i<s.size(); i++) sSum += s[i];
		for(int i=0; i<s.size(); i++) s[i] = s[i]/sSum;

		for(int i=0; i<z.size(); i++)
		{
			if(i==chosen) delta[i] = -diff*(1-s[i]);//*s[chosen];
			else 		  delta[i] =  diff*s[i];//*s[chosen];
		}

		for(int i=0; i<z.size(); i++)
		{
			if(z[i]-delta[i] > 0.9) delta[i] = z[i]-0.9;
			if(z[i]-delta[i] < 0.1) delta[i] = z[i]-0.1;
		}

		(*layers.rbegin())->setDelta(delta);

	}

	/*
	 *
	 */
	void NeuralNetwork::learnBackPropagation()
	{
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
	void NeuralNetwork::drawLayer(int layerNumber)
	{
		std::list<NNLayer*>::iterator layer = layers.begin();
		for(int i=0;i<layerNumber;i++) layer++;

		(*layer)->drawLayer();

	}

	/*
	 *
	 */
	void NeuralNetwork::saveToFile(std::string t_name)
	{
		std::ofstream file(t_name);

		file << layers.size() << ' ';
		for(auto it : layers)
		{
			it->saveToFile(file);
		}

		file.close();
	}

	/*
	 *
	 */
	void NeuralNetwork::loadFromFile(std::string t_name)
	{
		std::ifstream file(t_name);

		for(auto it = layers.begin(); it != layers.end(); it++)
		{
			delete (*it);
		}
		layers.clear();

		std::vector<NeuronsPtr> neuronPtrs; // TODO to list
		neuronPtrs.clear();

		int numberOfLayers,layerId;
		file >> numberOfLayers;

		while(file >> layerId)
		{
			NNLayer* layer;
			if(layerId == InputLayer::getLayerTypeId())
			{
				layer = InputLayer::loadFromFile(file);
				addLayer(layer);
			}
			else if(layerId == SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>::getLayerTypeId())
			{
				layer = SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>::loadFromFile(file,neuronPtrs);
				addLayer(layer);
			}
			else if(layerId == FuseLayer::getLayerTypeId())
			{
				layer = FuseLayer::loadFromFile(file,neuronPtrs);
				addLayer(layer);
			}
			else if(layerId == ConvolutionalLayer::getLayerTypeId())
			{
				layer = ConvolutionalLayer::loadFromFile(file,neuronPtrs);
				addLayer(layer);
			}
			else if(layerId == ConvSeparateWeightsLayer::getLayerTypeId())
			{
				layer = ConvSeparateWeightsLayer::loadFromFile(file,neuronPtrs);
				addLayer(layer);
			}
			else if(layerId == PoolingLayer::getLayerTypeId())
			{
				layer = PoolingLayer::loadFromFile(file,neuronPtrs);
				addLayer(layer);
			}
			neuronPtrs.push_back(layer->getNeuronPtr());
		}

		file.close();
	}

} /* namespace NeuralNetworkGPU */
