//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_SIGMOIDLAYER_H
#define NEURALNETWORKGPU_SIGMOIDLAYER_H

#include "NNLayer.h"

namespace NeuralNetworkGPU
{
	template<ActivationFunction F>
	class SigmoidLayer : public NNLayer
	{
	public:
		SigmoidLayer(float t_parameterB, float t_learnRate, int t_size, NeuronsPtr t_prevLayerReference);
		virtual ~SigmoidLayer();

	public:
		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void setDelta(std::vector<double> t_z) override;
		void learnSGD() override;
		void learnAdam() override;

		//configuration
		NeuronsPtr getNeuronPtr() override;
		void initWeights();
		void setWeights(float* t_weights);
		void setMomentum1(float* t_momentum);
		void setMomentum2(float* t_momentum);

		//save load
		void saveToFile(std::ofstream &t_file) override;
		static SigmoidLayer<F>* loadFromFile(std::ifstream &t_file, std::vector<NeuronsPtr> &t_prevLayerReferences);

		virtual void drawLayer();
		virtual void printInfo() override;

		static int getLayerTypeId() {return 1;}

	protected:
		float *de_input;
		int *d_inputSize, inputSize;

		float *d_output,*output;
		int size;
		int numberOfBlocks;
		int numberOfThreads;

		float *d_sums;
		float *d_weights;

		float *d_deltas, *deltas, *de_prevDeltas;

		float *d_n, *d_b;		// learning rate, b parameter
		float *d_m, *d_v;  	// 1st moment vector, 2nd moment vector
		float *d_B1, *d_B2; 	// Decay rates for moment vectors

		float learnRate;
	};
}

#endif //NEURALNETWORKGPU_SIGMOIDLAYER_H
