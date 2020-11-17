//
// Created by przemo on 27.12.2019.
//

#include "ConvolutionalLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	__global__ void determineOutputFuncConv(double *t_input, double *t_output, TensorSize *t_inputSize,
			double *t_sums,
			double *t_weights,
			MatrixSize *t_filterSize,
			double *t_deltas,
			double *d_b)
	{
		__shared__ double inputBuff[INPUT_BUFFER_SIZE];

		int inputSize = t_inputSize->x*t_inputSize->y*t_inputSize->z;
		//copy input to common buffer
		if(inputSize == blockDim.x)
		{
			inputBuff[threadIdx.x] = t_input[threadIdx.x];
		}
		else if(inputSize < blockDim.x)
		{
			if(threadIdx.x < inputSize) inputBuff[threadIdx.x] = t_input[threadIdx.x];
		}
		else if(inputSize > blockDim.x)
		{
			int index = inputSize-threadIdx.x-1;
			while(index >= 0)
			{
				inputBuff[index] = t_input[index];
				index -= blockDim.x;
			}
		}
		__syncthreads();

		long weightsIndex = inputSize*(threadIdx.x + blockIdx.x*blockDim.x);
		for(int x=0; x<t_filterSize->x; x++)
		{
			for(int y=0; y<t_filterSize->y; y++)
			{
				for(int z=0; z<t_inputSize->z; z++)
				{

				}
			}
		}

		//sums x[i]*w[i]
//		long weightsIndex = inputSize*(threadIdx.x + blockIdx.x*blockDim.x);
		double sum = t_weights[weightsIndex];
		for(int i=0; i<inputSize;i++)
		{
			sum += inputBuff[i] * t_weights[ weightsIndex+i+1 ];
		}
		t_sums[threadIdx.x + blockIdx.x*blockDim.x] = sum;
		//activation function
		t_output[threadIdx.x + blockIdx.x*blockDim.x] = 1 / (1 + exp(-(*d_b)*sum) );
		//reset delta
		t_deltas[threadIdx.x + blockIdx.x*blockDim.x] = 0;
	}

	/*
	 *
	 */
	__global__ void learnBackPropagationFuncConv(double *t_input, int *t_inputSize,
			double *t_output,
			double *t_sums,
			double *t_weights,
			double *t_deltas, double *t_prevDeltas,
			double *d_n,double *d_b)
	{
		int inputSize = *t_inputSize;

		//copy input to common buffer
		__shared__ double inputBuff[INPUT_BUFFER_SIZE];
		if(inputSize == blockDim.x)
		{
			inputBuff[threadIdx.x] = t_input[threadIdx.x];
		}
		else if(inputSize < blockDim.x)
		{
			if(threadIdx.x < inputSize) inputBuff[threadIdx.x] = t_input[threadIdx.x];
		}
		else if(inputSize > blockDim.x)
		{
			int index = inputSize-threadIdx.x-1;
			while(index >= 0)
			{
				inputBuff[index] = t_input[index];
				index -= blockDim.x;
			}
		}
		__syncthreads();

		long index = threadIdx.x +  blockIdx.x*blockDim.x;
		double delta = t_deltas[index];

		long weightsIndex = inputSize*(index);
		//determine common multiplier
		double e = exp(-(*d_b)*t_sums[index]);
		double m = 1 + e;
		double derivative = ((*d_b)*e/(m*m));

		double p = (*d_n)* delta * derivative;
		//calculate new weights
		//bias weight
		t_weights[weightsIndex] -= p;
		//rest weights
		for(int i=0; i<inputSize; i++)
		{
			t_weights[ weightsIndex+i+1 ] -= p*inputBuff[i];
		}

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			for(int i=0; i<*t_inputSize; i++)
			{
				int idx = weightsIndex + i + 1;
				t_prevDeltas[i] += delta * derivative * t_weights[idx] ;
			}
		}

		//reset delta
		t_deltas[index] = 0;

	}

	/*
	 *
	 */
	ConvolutionalLayer::ConvolutionalLayer(double t_parameterB, double t_learnRate,
			MatrixSize t_filterSize, TensorSize t_size, TensorSize t_inputSize, NeuronsPtr t_prevLayerReference)
	{
		size = t_size;
		de_input = t_prevLayerReference.inputPtr;

		//learn rate
		cudaMalloc( (void **) &d_n, sizeof(double));
		cudaMemcpy(d_n, &(t_learnRate), sizeof(double), cudaMemcpyHostToDevice);
		//parameter b
		cudaMalloc( (void **) &d_b, sizeof(double));
		cudaMemcpy(d_b, &(t_parameterB), sizeof(double), cudaMemcpyHostToDevice);

		//input size
		cudaMalloc( (void **) &d_inputSize, sizeof(TensorSize));
		cudaMemcpy(d_inputSize, &t_inputSize, sizeof(TensorSize), cudaMemcpyHostToDevice);
		inputSize = t_inputSize;

		//output
		cudaMalloc( (void **) &d_output, sizeof(double)*size.multiply());
		output = (double*) std::malloc(sizeof(double)*size.multiply());

		//weights
		cudaMalloc( (void **) &d_weights, sizeof(double)*size.multiply()*t_filterSize.multiply()*t_inputSize.z);
		initWeights();
		//filter size
		cudaMalloc( (void **) &d_filterSize, sizeof(MatrixSize));
		cudaMemcpy(d_filterSize, &t_filterSize, sizeof(MatrixSize), cudaMemcpyHostToDevice);

		//sums
		cudaMalloc( (void **) &d_sums, sizeof(double)*size.multiply());
		//deltas
		cudaMalloc( (void **) &d_deltas, sizeof(double)*size.multiply());
		deltas = (double*) malloc(sizeof(double)*size.multiply());
		de_prevDeltas = t_prevLayerReference.deltaPtr;

//		numberOfBlocks = 1;
//		while(1)
//		{
//			numberOfThreads = size/numberOfBlocks;
//			if(numberOfThreads<=800 && numberOfThreads*numberOfBlocks==size) break;
//			numberOfBlocks++;
//
//			assert(numberOfBlocks < 10 && "Could not match thread/block size");
//		}

	}

	/*
	 *
	 */
	ConvolutionalLayer::~ConvolutionalLayer()
	{
		cudaFree(d_n);
		cudaFree(d_b);

		cudaFree(d_inputSize);
		cudaFree(d_output);
		cudaFree(d_sums);
		cudaFree(d_weights);

		cudaFree(d_deltas);

		free(output);
		free(deltas);
	}

	/*
	 *
	 */
	void ConvolutionalLayer::initWeights()
	{
		long weightsSize = inputSize.multiply()*size.multiply()*inputSize.z;

		double *randomValues = (double*) malloc(sizeof(double)*weightsSize);

		for(int i=0; i< weightsSize; i++)
		{
//			std::cout << (int) (100*i/((inputSize+1)*size)) << "%\n";
			double randomValue = getRandomWeight();
			randomValues[i] = randomValue;

		}
		cudaMemcpy(d_weights, randomValues, sizeof(double)*weightsSize, cudaMemcpyHostToDevice);
		free(randomValues);
	}

	/*
	 *
	 */
	std::vector<double> ConvolutionalLayer::getOutput()
	{
		cudaMemcpy(output, d_output, sizeof(double)*size.multiply(), cudaMemcpyDeviceToHost);

		std::vector<double> result;
		int outputSize = size.multiply();
		for(int i=0; i<outputSize; i++ )
		{
			double v = output[i];
			result.push_back(v);
		}

		return result;
	}

	void ConvolutionalLayer::determineOutput()
	{
		determineOutputFuncConv<<< 12 , 12>>>(de_input, d_output, d_inputSize, d_sums, d_weights, d_filterSize, d_deltas, d_b);
	}

	void ConvolutionalLayer::setDelta(std::vector<double> t_z)
	{
		assert(t_z.size() == size.multiply() && "learning values size not match");

		#pragma omp parallel for shared(deltas,size,output, t_z) private(i) default(none)
		for(int i=0; i<size.multiply(); i++ )
		{
			deltas[i] = (double) output[i] - t_z[i];
		}

		cudaMemcpy(d_deltas, deltas, sizeof(double)*size.multiply(), cudaMemcpyHostToDevice);
	}

	void ConvolutionalLayer::learnBackPropagation()
	{
//		int64 timeBefore = cv::getTickCount();
//		learnBackPropagationFuncConv<<< numberOfThreads , numberOfBlocks >>>(de_input, d_inputSize, d_output, d_sums, d_weights, d_deltas, de_prevDeltas, d_n, d_b);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	/*
	 *
	 */
	NeuronsPtr ConvolutionalLayer::getNeuronPtr()
	{
		return NeuronsPtr(d_output,size.x*size.y*size.z, d_deltas);
	}
}
