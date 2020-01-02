//
// Created by przemo on 27.12.2019.
//

#ifndef QBOT_NNLAYER_H
#define QBOT_NNLAYER_H

#include <cassert>
#include <vector>
#include "../Neuron/Neuron.h"

class NNLayer
{
public:

    virtual ~NNLayer() = default;

    //input
    virtual void setInput(std::vector<int> t_input) = 0;
    virtual void setInput(std::vector<double> t_input) = 0;

    //output
    virtual std::vector<double> getOutput() = 0;
    virtual void determineOutput() = 0;

    //learn
    virtual void setDelta(std::vector<double> t_z) {};
    virtual void learnBackPropagation() = 0;
    virtual void calculateDerivative() {}

    //configuration
    virtual std::vector<Neuron*> getNeuronPtr() = 0;

};

#endif //QBOT_NNLAYER_H
