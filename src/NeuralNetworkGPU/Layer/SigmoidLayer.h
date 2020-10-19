//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_SIGMOIDLAYER_H
#define NEURALNETWORKGPU_SIGMOIDLAYER_H

#include "NNLayer.h"

namespace NeuralNetworkGPU
{
	class SigmoidLayer : public NNLayer
	{
	public:
		SigmoidLayer(double t_parameterB, double t_learnRate, int t_size, NeuronsPtr t_prevLayerReference);
		virtual ~SigmoidLayer();

	public:
		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void setDelta(std::vector<double> t_z) override;
		void learnBackPropagation() override;

		//configuration
		NeuronsPtr getNeuronPtr() override;
		void initWeights();

		//save load
//		void saveToFile(std::ofstream &t_file) override;
//		void loadFromFile(std::ifstream &t_file) override;

	protected:
		double *de_input;
		int *d_inputSize, inputSize;

		double *d_output,*output;
		int size;
		int numberOfBlocks;
		int numberOfThreads;

		double *d_sums;
		double *d_weights;

		double *d_deltas, *deltas, *de_prevDeltas;

		double *d_n, *d_b;

		double learnRate;
		static double b;
	};
}

#endif //NEURALNETWORKGPU_SIGMOIDLAYER_H
