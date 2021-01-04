//
// Created by przemo on 27.12.2019.
//

#ifndef QBOT_SIGMOIDLAYER_H
#define QBOT_SIGMOIDLAYER_H

#include "NNLayer.h"
#include "../Neuron/InputNeuron.h"
#include "../Neuron/AdaptiveNeuron.h"

namespace NeuralNetworkCPU
{
	class SigmoidLayer : public NNLayer
	{
	public:
		SigmoidLayer(double t_parameterB, double t_learnRate, int t_size, std::vector<Neuron*> t_prevLayerReference);
		virtual ~SigmoidLayer() = default;

	public:
		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void setDelta(std::vector<double> t_z) override;
		void learnSGD() override;

		//configuration
		std::vector<Neuron*> getNeuronPtr() override;
		static void configure(double t_activationFunctionParameter) {b = t_activationFunctionParameter;}

		//save load
		void saveToFile(std::ofstream &t_file) override;
		void loadFromFile(std::ifstream &t_file) override;

	protected:
		std::vector<AdaptiveNeuron> neurons;
		InputNeuron biasValue;

		double learnRate;

		static double b;
	};
}

#endif //QBOT_SIGMOIDLAYER_H
