//
// Created by przemo on 27.12.2019.
//

#include "InputLayer.h"

/*
 *
 */
InputLayer::InputLayer(int t_size)
{
    for(int i=0; i < t_size; i++)
    {
        neurons.push_back(InputNeuron());
    }
}

/*
 *
 */
void InputLayer::setInput(std::vector<int> t_input)
{
    assert(t_input.size() == neurons.size() && "InputLayer::setInput input size not match");
    for(int i=0; i<neurons.size(); i++ )
    {
        neurons[i].setValue(t_input[i]);
    }
}

/*
 *
 */
void InputLayer::setInput(std::vector<double> t_input)
{
    assert(t_input.size() == neurons.size() && "InputLayer::setInput input size not match");
    for(int i=0; i<neurons.size(); i++ )
    {
        neurons[i].setValue(t_input[i]);
    }
}

/*
 *
 */
std::vector<double> InputLayer::getOutput()
{
    std::vector<double> result;
    for( auto it = neurons.begin(); it != neurons.end(); it++)
    {
        result.push_back(it->getOutput());
    }
    return result;
}

/*
 *
 */
void InputLayer::determineOutput()
{
    //Do nothing
}

/*
 *
 */
void InputLayer::learnBackPropagation()
{
    //Do nothing
}

/*
 *
 */
std::vector<Neuron *> InputLayer::getNeuronPtr()
{
    std::vector<Neuron *> result;
    for( auto it = neurons.begin(); it != neurons.end(); it++)
    {
        result.push_back(&(*it));
    }

    return result;
}
