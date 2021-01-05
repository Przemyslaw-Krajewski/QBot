//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_CONVOLUTIONALLAYER_H
#define NEURALNETWORKGPU_CONVOLUTIONALLAYER_H

#include "NNLayer.h"

namespace NeuralNetworkGPU
{
	class ConvolutionalLayer : public NNLayer
	{
	public:
		ConvolutionalLayer(float t_parameterB, float t_learnRate, MatrixSize t_filterSize, TensorSize t_size, TensorSize t_inputSize, NeuronsPtr t_prevLayerReference);;
		virtual ~ConvolutionalLayer();

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

		//save load
//		void saveToFile(std::ofstream &t_file) override;
//		void loadFromFile(std::ifstream &t_file) override;

	protected:
		float *de_input;
		TensorSize *d_inputSize,inputSize;

		float *d_output,*output;
		TensorSize size;

		float *d_sums;
		float *d_weights;
		MatrixSize *d_filterSize;

		float *d_deltas, *deltas, *de_prevDeltas;

		float *d_n, *d_b;
	};
}

#endif //NEURALNETWORKGPU_CONVOLUTIONALLAYER_H
