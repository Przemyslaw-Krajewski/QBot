//
// Created by przemo on 27.12.2019.
//

#include "SigmoidLayer.h"

namespace NeuralNetworkCPU
{

	double SigmoidLayer::b = 0;

	/*
	 *
	 */
	SigmoidLayer::SigmoidLayer(double t_parameterB, double t_learnRate, int t_size, std::vector<Neuron *> t_prevLayerReference)
	{
		biasValue = InputNeuron(1.0);
		t_prevLayerReference.push_back(&biasValue);

		learnRate = t_learnRate;

		for(int i=0; i < t_size; i++)
		{
			neurons.push_back(AdaptiveNeuron(t_prevLayerReference, &learnRate,
					 [t_parameterB](double x) -> double { return 1 / ( 1 + exp(-t_parameterB* x) ); },
					[t_parameterB](double x) -> double { double e = exp(-t_parameterB*x);
						  double m = 1 + e;
						  return (t_parameterB*e/(m*m));}));
		}
	}

	/*
	 *
	 */
	std::vector<double> SigmoidLayer::getOutput()
	{
		std::vector<double> result;
		for( auto it = neurons.begin(); it != neurons.end(); it++)
		{
			result.push_back(it->getOutput());
		}
		return result;
	}

	void SigmoidLayer::determineOutput()
	{
		int i;
		#pragma omp parallel for shared(neurons) private(i) default(none)
		for(i=0; i<neurons.size(); i++)
		{
			neurons[i].determineOutput();
		}
	}

	void SigmoidLayer::setDelta(std::vector<double> t_z)
	{
		assert(t_z.size() == neurons.size() && "learning values size not match");
		int i=0;
		for( auto it = neurons.begin(); it != neurons.end(); it++,i++)
		{
			it->setDelta(-t_z[i]+it->getOutput());
		}
	}

	void SigmoidLayer::learnSGD()
	{

	//	int64 timeBefore = cv::getTickCount();
		int i;
		#pragma omp parallel for shared(neurons) private(i) default(none)
		for(i=0; i<neurons.size(); i++)
		{
			neurons[i].learnSGD();
		}
	//	int64 afterBefore = cv::getTickCount();
	//	std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	/*
	 *
	 */
	std::vector<Neuron *> SigmoidLayer::getNeuronPtr()
	{
		std::vector<Neuron *> result;
		for( auto it = neurons.begin(); it != neurons.end(); it++)
		{
			result.push_back(&(*it));
		}

		return result;
	}

	/*
	 *
	 */
	void SigmoidLayer::saveToFile(std::ofstream & t_file)
	{
		t_file << (double) 1 << ' '; //Signature of SigmoidLayer
		t_file << (double) neurons.size() << ' ';
		t_file << learnRate << ' ';
		t_file << b << ' ';

		for( auto it = neurons.begin(); it != neurons.end(); it++)
		{
			std::vector<double> *weights = it->getWeights();
			for(int i=0; i<weights->size(); i++)
			{
				t_file << (*weights)[i] << ' ';
			}
		}
	}

	/*
	 *
	 */
	void SigmoidLayer::loadFromFile(std::ifstream & t_file)
	{
		for( auto it = neurons.begin(); it != neurons.end(); it++)
		{
			double buff;
			std::vector<double> *weights = it->getWeights();
			for(int i=0; i<weights->size(); i++)
			{
				if(t_file.eof()) {assert("SigmoidLayer LoadFromFile: unexpected end of file");}
				t_file >> buff;
				(*weights)[i] = buff;
			}
		}
	}
}
