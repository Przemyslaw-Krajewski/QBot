//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_NEURON_LAYER_CU_
#define NEURALNETWORKGPU_NEURON_LAYER_CU_

#include <cassert>
#include <vector>

#include <opencv2/opencv.hpp>

#define INPUT_BUFFER_SIZE 4096

namespace NeuralNetworkGPU
{
	struct TensorSize
	{
		TensorSize(const TensorSize& t_ts) {x = t_ts.x;y = t_ts.y;z = t_ts.z;}
		TensorSize(int t_x, int t_y, int t_z) { x=t_x;y=t_y;z=t_z;}
		TensorSize() : TensorSize(0,0,0) {};
		int multiply() { return x*y*z;}
		int x,y,z;
	};

	struct MatrixSize
	{
		MatrixSize(const MatrixSize& t_ms) {x = t_ms.x;y = t_ms.y;}
		MatrixSize(int t_x, int t_y) { x=t_x;y=t_y;}
		int multiply() { return x*y;}
		int x,y;
	};

	struct NeuronsPtr
	{
		NeuronsPtr(float* t_inputPtr, int t_size, float* t_deltaPtr) {inputPtr = t_inputPtr; size = t_size; deltaPtr = t_deltaPtr;}
		float* inputPtr;
		float* deltaPtr;
		int size;
	};

	class NNLayer
	{
	public:

		NNLayer() {};
		virtual ~NNLayer() = default;

		//input
		virtual void setInput(std::vector<int> t_input) {};
		virtual void setInput(std::vector<double> t_input) {};

		//output
		virtual std::vector<double> getOutput() = 0;
		virtual void determineOutput() = 0;

		//learn
		virtual void setDelta(std::vector<double> t_z) {};
		virtual void learnSGD() = 0;
		virtual void learnAdam() = 0;

		//configuration
		virtual NeuronsPtr getNeuronPtr() = 0;
		virtual TensorSize getTensorOutputSize() {assert("getTensorOutputSize() Not implemented" && 0); return TensorSize(0,0,0);}

		//visualization
		virtual void drawLayer() {};

		//save load
		virtual void saveToFile(std::ofstream &t_file) {assert("saveToFile() Not implemented" && 0);}
		virtual void loadFromFile(std::ifstream &t_file) {assert("loadFromFile() Not implemented" && 0);}

	public:
		static double getRandomWeight() { return 0.25*((double)((rand()%100000))/100000-0.5); }

	};
}

#endif //QBOT_NNLAYER_H
