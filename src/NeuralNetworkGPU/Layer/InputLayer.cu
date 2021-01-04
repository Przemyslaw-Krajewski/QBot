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
//				int *d_a, *d_b;
//				int a=15;int b;
//				cudaMalloc( (void **) &d_a, sizeof(int));
//				cudaMalloc( (void **) &d_b, sizeof(int));
//
//				cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
//				funkcja<<<2,1>>>(d_a,d_b);
//				cudaMemcpy(&b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
//
//				std::cout << a << "  " << b << "\n";
//
//				cudaFree(d_a);
//				cudaFree(d_b);
//
//				InputNeuron *d_n;
//				cudaMalloc( (void **) &d_n, sizeof(InputNeuron));
//
//				double *d_in, *d_out;
//				double in=2.5;double out;
//				cudaMalloc( (void **) &d_in, sizeof(double));
//				cudaMalloc( (void **) &d_out, sizeof(double));
//
//				cudaMemcpy(d_in, &in, sizeof(double), cudaMemcpyHostToDevice);
//				funkcja2<<<1,1>>>(d_n,d_in,d_out);
//				cudaMemcpy(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
//				std::cout << out << "\n";
//
//				cudaFree(d_n);
//				cudaFree(d_in);
//				cudaFree(d_out);
		if(INPUT_BUFFER_SIZE < t_size)
		{
			std::cout << t_size;
			assert("CUDA input buffer to small");
		}
		size = t_size;
		input = (double*) malloc(sizeof(double)*size);
		cudaMalloc( (void **) &d_input, sizeof(double)*size);

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
			input[i] = (double) t_input[i];
		}

		cudaMemcpy(d_input, input, sizeof(double)*size, cudaMemcpyHostToDevice);

	}

	/*
	 *
	 */
	void InputLayer::setInput(std::vector<double> t_input)
	{
		assert(t_input.size() == size && "InputLayer::setInput input size not match");

		for(int i=0; i<size; i++ )
		{
			input[i] = (double) t_input[i];
		}

		cudaMemcpy(d_input, input, sizeof(double)*size, cudaMemcpyHostToDevice);

	}

	/*
	 *
	 */
	std::vector<double> InputLayer::getOutput()
	{
		cudaMemcpy(input, d_input, sizeof(double)*size, cudaMemcpyDeviceToHost);

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
		return NeuronsPtr(d_input,size, nullptr);
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
