//
// Created by przemo on 27.12.2019.
//

#ifndef QBOT_INPUTLAYER_H
#define QBOT_INPUTLAYER_H

#include "NNLayer.h"
#include "../Neuron/InputNeuron.h"

namespace NeuralNetworkCPU
{
	class InputLayer : public NNLayer
	{
	public:
		InputLayer(int t_size);
		virtual ~InputLayer() = default;

	public:
		//input
		void setInput(std::vector<int> t_input) override;
		void setInput(std::vector<double> t_input) override;

		//output
		std::vector<double> getOutput() override;
		void determineOutput() override;

		//learn
		void learnSGD() override ;

		//configuration
		std::vector<Neuron*> getNeuronPtr() override;

		//save load
		void saveToFile(std::ofstream &t_file) override;

	protected:
		std::vector<InputNeuron> neurons;
	};
}


#endif //QBOT_INPUTLAYER_H
