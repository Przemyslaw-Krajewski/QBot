//
// Created by przemo on 27.12.2019.
//

#ifndef QBOT_NNLAYER_H
#define QBOT_NNLAYER_H

#include <cassert>
#include <vector>
#include "../Neuron/Neuron.h"

#include <opencv2/opencv.hpp>

struct TensorSize
{
	TensorSize(const TensorSize& t_ts) {x = t_ts.x;y = t_ts.y;z = t_ts.z;}
	TensorSize(int t_x, int t_y, int t_z) { x=t_x;y=t_y;z=t_z;}
	int x,y,z;
};

struct MatrixSize
{
	MatrixSize(const MatrixSize& t_ms) {x = t_ms.x;y = t_ms.y;}
	MatrixSize(int t_x, int t_y) { x=t_x;y=t_y;}
	int x,y;
};

class NNLayer
{
public:

    virtual ~NNLayer() = default;

    //input
    virtual void setInput(std::vector<int> t_input) {};
    virtual void setInput(std::vector<double> t_input) {};

    //output
    virtual std::vector<double> getOutput() = 0;
    virtual void determineOutput() = 0;

    //learn
    virtual void setDelta(std::vector<double> t_z) {};
    virtual void learnBackPropagation() = 0;

    //configuration
    virtual std::vector<Neuron*> getNeuronPtr() = 0;
    virtual TensorSize getTensorOutputSize() {assert("Not implemented" && 0);}

    //visualization
    virtual void drawLayer() {};

};

#endif //QBOT_NNLAYER_H
