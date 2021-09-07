//
// Created by przemo on 27.12.2019.
//

#include "InputLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	InputLayer::InputLayer(int t_size)
	{
		assert(INPUT_BUFFER_SIZE >= t_size && "CUDA input buffer to small");

		size = t_size;
		input = (float*) malloc(sizeof(float)*size);
		cudaMalloc( (void **) &d_input, sizeof(float)*size);

	}

	/*
	 *
	 */
	InputLayer::InputLayer(TensorSize t_size)
	{
		size = t_size.m;
		tSize = t_size;
		input = (float*) malloc(sizeof(float)*tSize.m);
		cudaMalloc( (void **) &d_input, sizeof(float)*tSize.m);

	}

	/*
	 *
	 */
	InputLayer::~InputLayer()
	{
		cudaFree(d_input);
		free(input);
	}

	/*
	 *
	 */
	void InputLayer::setInput(std::vector<int> t_input)
	{
		assert(t_input.size() == size && "InputLayer::setInput input size not match");

		#pragma omp parallel for shared(input, t_input, size) private(i) default(none)
		for(int i=0; i<size; i++ )
		{
			input[i] = (float) t_input[i];
		}

		cudaMemcpy(d_input, input, sizeof(float)*size, cudaMemcpyHostToDevice);

	}

	/*
	 *
	 */
	void InputLayer::setInput(std::vector<double> t_input)
	{

		if(tSize.m > 0)
		{
			assert(t_input.size()==tSize.m && "InputLayer::setInput input size not match");

			for(int i=0; i<tSize.m; i++ )
			{
				input[i] = (float) t_input[i];
			}
			cudaMemcpy(d_input, input, sizeof(float)*tSize.m, cudaMemcpyHostToDevice);
		}
		else
		{
			assert(t_input.size() == size && "InputLayer::setInput input size not match");

			for(int i=0; i<size; i++ )
			{
				input[i] = (float) t_input[i];
			}
			cudaMemcpy(d_input, input, sizeof(float)*size, cudaMemcpyHostToDevice);
		}

	}

	/*
	 *
	 */
	std::vector<double> InputLayer::getOutput()
	{
		cudaMemcpy(input, d_input, sizeof(float)*size, cudaMemcpyDeviceToHost);

		std::vector<double> result;
		for(int i=0; i<size; i++ )
		{
			result.push_back(input[i]);
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
	void InputLayer::learnSGD()
	{
		//Do nothing
	}

	/*
	 *
	 */
	void InputLayer::learnAdam()
	{
		//Do nothing
	}

	/*
	 *
	 */
	NeuronsPtr InputLayer::getNeuronPtr()
	{
		if(tSize.m != 0) return NeuronsPtr(layerId, d_input,tSize, nullptr);
		else return NeuronsPtr(layerId, d_input,size, nullptr);
	}

	/*
	 *
	 */
	void InputLayer::saveToFile(std::ofstream &t_file)
	{
		t_file << (float) getLayerTypeId() << ' ';
		t_file << (float) size << ' ';
		t_file << (float) tSize.x << ' ';
		t_file << (float) tSize.y << ' ';
		t_file << (float) tSize.z << ' ';
	}

	/*
	 *
	 */
	InputLayer* InputLayer::loadFromFile(std::ifstream & t_file)
	{
		float size, tSize[3];
		t_file >> size;
		t_file >> tSize[0];
		t_file >> tSize[1];
		t_file >> tSize[2];

		if(tSize[0] == 0 || tSize[1] == 0 || tSize[2] == 0)
		{
			return new NeuralNetworkGPU::InputLayer(size);
		}
		else
		{
			return new NeuralNetworkGPU::InputLayer(TensorSize(tSize[0],tSize[1],tSize[2]));
		}

	}
}
