//
// Created by przemo on 27.12.2019.
//

#ifndef QBOT_INPUTLAYER_H
#define QBOT_INPUTLAYER_H

#include "NNLayer.h"
#include "../Neuron/InputNeuron.h"

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
    void learnBackPropagation() override ;

    //configuration
    std::vector<Neuron*> getNeuronPtr() override;

protected:
    std::vector<InputNeuron> neurons;
};


#endif //QBOT_INPUTLAYER_H
