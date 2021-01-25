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
//		funkcja3<<<1,size>>>(d_input);
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
		if(tSize.m != 0) return NeuronsPtr(d_input,tSize, nullptr);
		else return NeuronsPtr(d_input,size, nullptr);
	}

	/*
	 *
	 */
//	void InputLayer::saveToFile(std::ofstream &t_file)
//	{
//		t_file << (double) 0 << ' '; //Signature of InputLayer
//		t_file << (double) neurons.size() << ' ';
//	}
}
