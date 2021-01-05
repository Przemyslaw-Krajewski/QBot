//
// Created by przemo on 27.12.2019.
//

#include "ConvolutionalLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	__global__ void determineOutputFuncConv(float *t_input, float *t_output, TensorSize *t_inputSize,
			float *t_sums,
			float *t_weights,
			MatrixSize *t_filterSize,
			float *t_deltas,
			float *d_b)
	{
		__shared__ float inputBuff[INPUT_BUFFER_SIZE];

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
		float sum = t_weights[weightsIndex];
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
	__global__ void learnBackPropagationFuncConv(float *t_input, int *t_inputSize,
			float *t_output,
			float *t_sums,
			float *t_weights,
			float *t_deltas, float *t_prevDeltas,
			float *d_n,float *d_b)
	{
		int inputSize = *t_inputSize;

		//copy input to common buffer
		__shared__ float inputBuff[INPUT_BUFFER_SIZE];
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
		float delta = t_deltas[index];

		long weightsIndex = inputSize*(index);
		//determine common multiplier
		float e = exp(-(*d_b)*t_sums[index]);
		float m = 1 + e;
		float derivative = ((*d_b)*e/(m*m));

		float p = (*d_n)* delta * derivative;
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
	ConvolutionalLayer::ConvolutionalLayer(float t_parameterB, float t_learnRate,
			MatrixSize t_filterSize, TensorSize t_size, TensorSize t_inputSize, NeuronsPtr t_prevLayerReference)
	{
		size = t_size;
		de_input = t_prevLayerReference.inputPtr;

		//learn rate
		cudaMalloc( (void **) &d_n, sizeof(float));
		cudaMemcpy(d_n, &(t_learnRate), sizeof(float), cudaMemcpyHostToDevice);
		//parameter b
		cudaMalloc( (void **) &d_b, sizeof(float));
		cudaMemcpy(d_b, &(t_parameterB), sizeof(float), cudaMemcpyHostToDevice);

		//input size
		cudaMalloc( (void **) &d_inputSize, sizeof(TensorSize));
		cudaMemcpy(d_inputSize, &t_inputSize, sizeof(TensorSize), cudaMemcpyHostToDevice);
		inputSize = t_inputSize;

		//output
		cudaMalloc( (void **) &d_output, sizeof(float)*size.multiply());
		output = (float*) std::malloc(sizeof(float)*size.multiply());

		//weights
		cudaMalloc( (void **) &d_weights, sizeof(float)*size.multiply()*t_filterSize.multiply()*t_inputSize.z);
		initWeights();
		//filter size
		cudaMalloc( (void **) &d_filterSize, sizeof(MatrixSize));
		cudaMemcpy(d_filterSize, &t_filterSize, sizeof(MatrixSize), cudaMemcpyHostToDevice);

		//sums
		cudaMalloc( (void **) &d_sums, sizeof(float)*size.multiply());
		//deltas
		cudaMalloc( (void **) &d_deltas, sizeof(float)*size.multiply());
		deltas = (float*) malloc(sizeof(float)*size.multiply());
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

		float *randomValues = (float*) malloc(sizeof(float)*weightsSize);

		for(int i=0; i< weightsSize; i++)
		{
//			std::cout << (int) (100*i/((inputSize+1)*size)) << "%\n";
			float randomValue = getRandomWeight();
			randomValues[i] = randomValue;

		}
		cudaMemcpy(d_weights, randomValues, sizeof(float)*weightsSize, cudaMemcpyHostToDevice);
		free(randomValues);
	}

	/*
	 *
	 */
	std::vector<double> ConvolutionalLayer::getOutput()
	{
		cudaMemcpy(output, d_output, sizeof(float)*size.multiply(), cudaMemcpyDeviceToHost);

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
			deltas[i] = (float) output[i] - t_z[i];
		}

		cudaMemcpy(d_deltas, deltas, sizeof(float)*size.multiply(), cudaMemcpyHostToDevice);
	}

	void ConvolutionalLayer::learnSGD()
	{
//		int64 timeBefore = cv::getTickCount();
//		learnBackPropagationFuncConv<<< numberOfThreads , numberOfBlocks >>>(de_input, d_inputSize, d_output, d_sums, d_weights, d_deltas, de_prevDeltas, d_n, d_b);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	void ConvolutionalLayer::learnAdam()
	{
		//Implement SGD first
	}

	/*
	 *
	 */
	NeuronsPtr ConvolutionalLayer::getNeuronPtr()
	{
		return NeuronsPtr(d_output,size.x*size.y*size.z, d_deltas);
	}
}
