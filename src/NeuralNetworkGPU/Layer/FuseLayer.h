//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_FUSELAYER_H
#define NEURALNETWORKGPU_FUSELAYER_H

#include "NNLayer.h"

namespace NeuralNetworkGPU
{
	class FuseLayer : public NNLayer
	{
	public:
		FuseLayer(NeuronsPtr t_prevLayerReference1, NeuronsPtr t_prevLayerReference2);
		virtual ~FuseLayer();

	public:
		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void learnSGD() override;
		void learnAdam() override;

		//configuration
		NeuronsPtr getNeuronPtr() override;

		//save load
		void saveToFile(std::ofstream &t_file) override;
		static FuseLayer* loadFromFile(std::ifstream &t_file, std::vector<NeuronsPtr> &t_prevLayerReferences);

		virtual void drawLayer() {std::cout << "FuseLayer::drawLayer() not implemented\n";};
		virtual void printInfo() override;

		static int getLayerTypeId() {return 2;}

	protected:
		float *de_input1,*de_input2;
		int *d_inputSize1, inputSize1;
		int *d_inputSize2, inputSize2;

		float *d_output,*output;
		int size;
		int numberOfBlocks;
		int numberOfThreads;

		float *d_deltas, *deltas, *de_prevDeltas1, *de_prevDeltas2;

		int idFusedLayer1,idFusedLayer2;
	};
}

#endif //NEURALNETWORKGPU_FUSELAYER_H
