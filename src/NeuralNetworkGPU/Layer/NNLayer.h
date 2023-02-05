//
// Created by przemo on 27.12.2019.
//

#ifndef NEURALNETWORKGPU_NEURON_LAYER_CU_
#define NEURALNETWORKGPU_NEURON_LAYER_CU_

#define GPU_NN_STRUCTURE_LOG

#include <cassert>
#include <vector>
#include <functional>
#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>

#define INPUT_BUFFER_SIZE 6480

namespace NeuralNetworkGPU
{
	enum class ActivationFunction {Linear, Sigmoid, RELU, LeakRELU};

	struct TensorSize
	{
		TensorSize(const TensorSize& t_ts) 
		{
			x = t_ts.x;
			y = t_ts.y;
			z = t_ts.z;
			m = x*y*z;
		}
		TensorSize(int t_x, int t_y, int t_z) 
		{
			x=t_x;
			y=t_y;
			z=t_z;
			m = x*y*z;
		}
		TensorSize() : TensorSize(0,0,0) {}
		int multiply() { m=x*y*z; return m;}
		int x,y,z;
		int m;
	};

	struct MatrixSize
	{
		MatrixSize(const MatrixSize& t_ms) 
		{
			x = t_ms.x;
			y = t_ms.y;
			m = x*y;
		}
		MatrixSize(int t_x, int t_y)
		{
			x=t_x;
			y=t_y;
			m = x*y;
		}
		MatrixSize() : MatrixSize(0,0) {}
		int multiply() { m=x*y; return m;}
		int x,y;
		int m;
	};

	struct NeuronsPtr
	{
		NeuronsPtr(int t_id, float* t_inputPtr, int t_size, float* t_deltaPtr)
		{
			inputPtr = t_inputPtr;
			size = t_size;
			deltaPtr = t_deltaPtr;
			tSize = TensorSize(-1,-1,-1);
			id = t_id;
		}
		NeuronsPtr(int t_id, float* t_inputPtr, TensorSize t_tSize, float* t_deltaPtr)
		{
			inputPtr = t_inputPtr;
			size = t_tSize.m;
			deltaPtr = t_deltaPtr;
			tSize = t_tSize;
			id = t_id;
		}
		float* inputPtr;
		float* deltaPtr;
		int size;
		TensorSize tSize;
		int id;
	};

	class NNLayer
	{
	public:

		NNLayer() : randomNumberGenerator(std::mt19937(randomDevice())) {};
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
		virtual TensorSize getTensorOutputSize() {assert("NNLayer::getTensorOutputSize() Not implemented" && 0); return TensorSize(0,0,0);}

		//visualization
		virtual void drawLayer() {std::cout << "NNLayer::drawLayer() not implemented\n";}
		virtual void printInfo() {std::cout << "	(" << layerId << ") NNLayer <-- " << prevLayerId << "\n";}

		//save load
		virtual void saveToFile(std::ofstream &t_file) {assert("saveToFile() Not implemented" && 0);}

		//Layer Id used for load and save
		int getLayerId() {return layerId;}
		void setLayerId(int id) {layerId = id;}

	public:
		double getRandomWeight() { return std::uniform_real_distribution<>(-0.1, 0.1)(randomNumberGenerator);}

	protected:
		int layerId{-1};
		int prevLayerId{-1};
	private:
	    std::random_device randomDevice;
	    std::mt19937 randomNumberGenerator;
	};
}

#endif //QBOT_NNLAYER_H
