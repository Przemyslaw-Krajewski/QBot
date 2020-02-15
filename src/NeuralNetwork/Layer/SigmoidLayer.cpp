//
// Created by przemo on 27.12.2019.
//

#include "SigmoidLayer.h"

double SigmoidLayer::b = 0;

/*
 *
 */
SigmoidLayer::SigmoidLayer(double t_learnRate, int t_size, std::vector<Neuron *> t_prevLayerReference)
{
    biasValue = InputNeuron(1.0);
    t_prevLayerReference.push_back(&biasValue);

    learnRate = t_learnRate;

    for(int i=0; i < t_size; i++)
    {
        neurons.push_back(AdaptiveNeuron(t_prevLayerReference, &learnRate,
                 [](double x) -> double { return 1 / ( 1 + exp(-b* x) ); },
                [](double x) -> double { double e = exp(-b*x);
                      double m = 1 + e;
                      return -(b*e/(m*m));}));
    }
}

/*
 *
 */
std::vector<double> SigmoidLayer::getOutput()
{
    std::vector<double> result;
    for( auto it = neurons.begin(); it != neurons.end(); it++)
    {
        result.push_back(it->getOutput());
    }
    return result;
}

void SigmoidLayer::determineOutput()
{
    int i;
    #pragma omp parallel for shared(neurons) private(i) default(none)
    for(i=0; i<neurons.size(); i++)
    {
        neurons[i].determineOutput();
    }
}

void SigmoidLayer::setDelta(std::vector<double> t_z)
{
    assert(t_z.size() == neurons.size() && "learning values size not match");
    int i=0;
    for( auto it = neurons.begin(); it != neurons.end(); it++,i++)
    {
        it->setDelta(t_z[i]-it->getOutput());
    }
}

void SigmoidLayer::learnBackPropagation()
{

//	int64 timeBefore = cv::getTickCount();
    int i;
    #pragma omp parallel for shared(neurons) private(i) default(none)
    for(i=0; i<neurons.size(); i++)
    {
        neurons[i].learnDeltaRule();
    }
//	int64 afterBefore = cv::getTickCount();
//	std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
}

/*
 *
 */
std::vector<Neuron *> SigmoidLayer::getNeuronPtr()
{
    std::vector<Neuron *> result;
    for( auto it = neurons.begin(); it != neurons.end(); it++)
    {
        result.push_back(&(*it));
    }

    return result;
}