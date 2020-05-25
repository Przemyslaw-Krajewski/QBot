//
// Created by przemo on 02.01.2020.
//

#include "PoolingLayer.h"

namespace NeuralNetworkCPU
{
	/*
	 *
	 */
	PoolingLayer::PoolingLayer(MatrixSize t_outputLayerSize, TensorSize t_inputSize, std::vector<Neuron*> t_prevLayerReference)
		: outputSize(t_outputLayerSize.x, t_outputLayerSize.y, t_inputSize.z)
	{
		int xRatio = t_inputSize.x/t_outputLayerSize.x;
		int yRatio = t_inputSize.y/t_outputLayerSize.y;

		for(int z=0; z<t_inputSize.z; z++)
		{
			for(int y=0; y+yRatio-1<t_inputSize.y; y+=yRatio)
			{
				for (int x=0; x+xRatio-1<t_inputSize.x; x+=xRatio)
				{
					std::vector<Neuron*> neuronsReference;
					for (int fy=0; fy < yRatio; fy++)
					{
						for (int fx=0; fx < xRatio; fx++)
						{
							neuronsReference.push_back(t_prevLayerReference[getIndex(x + fx, y + fy, z, t_inputSize.x,t_inputSize.y)]);
						}
					}

					neurons.push_back(PoolingNeuron(neuronsReference));
				}
			}
		}
	}

	/*
	 *
	 */
	std::vector<double> PoolingLayer::getOutput()
	{
		std::vector<double> result;
		for( auto it = neurons.begin(); it != neurons.end(); it++)
		{
			result.push_back(it->getOutput());
		}
		return result;
	}

	void PoolingLayer::determineOutput()
	{
		int i;
		#pragma omp parallel for shared(neurons) private(i) default(none)
		for(i=0; i<neurons.size(); i++)
		{
			neurons[i].determineOutput();
		}
	}


	/*
	 *
	 */
	void PoolingLayer::setDelta(std::vector<double> t_z)
	{
		assert(t_z.size() == neurons.size() && "learning values size not match");
		int i=0;
		for( auto it = neurons.begin(); it != neurons.end(); it++,i++)
		{
			it->setDelta(t_z[i]-it->getOutput());
		}
	}

	/*
	 *
	 */
	void PoolingLayer::learnBackPropagation()
	{
	//	int64 timeBefore = cv::getTickCount();
		int i;
	#pragma omp parallel for shared(neurons) private(i) default(none)
		for(i=0; i<neurons.size(); i++)
		{
			neurons[i].learnDeltaRule();
		}
	//	int64 afterBefore = cv::getTickCount();
	//	std::cout << "Pool: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	/*
	 *
	 */
	std::vector<Neuron *> PoolingLayer::getNeuronPtr()
	{
		std::vector<Neuron *> result;
		for( auto it = neurons.begin(); it != neurons.end(); it++)
		{
			result.push_back(&(*it));
		}

		return result;
	}
}
