//
// Created by przemo on 27.12.2019.
//

#include "SigmoidLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	__global__ void determineOutputFunc(double *t_input, double *t_output, int *t_inputSize,
			double *t_sums,
			double *t_weights,
			double *t_deltas,
			double *d_b)
	{
		int inputSize = (*t_inputSize);
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

		//sums x[i]*w[i]
		long weightsIndex = inputSize*(threadIdx.x + blockIdx.x*blockDim.x);
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
	__global__ void learnBackPropagationFunc(double *t_input, int *t_inputSize,
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

	double SigmoidLayer::b = 0;

	/*
	 *
	 */
	SigmoidLayer::SigmoidLayer(double t_parameterB, double t_learnRate, int t_size, NeuronsPtr t_prevLayerReference)
	{
		size = t_size;
		de_input = t_prevLayerReference.inputPtr;

		cudaMalloc( (void **) &d_n, sizeof(double));
		cudaMemcpy(d_n, &(t_learnRate), sizeof(double), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_b, sizeof(double));
		cudaMemcpy(d_b, &(t_parameterB), sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc( (void **) &d_inputSize, sizeof(int));
		cudaMemcpy(d_inputSize, &(t_prevLayerReference.size), sizeof(int), cudaMemcpyHostToDevice);
		inputSize = t_prevLayerReference.size;

		cudaMalloc( (void **) &d_output, sizeof(double)*size);
		output = (double*) std::malloc(sizeof(double)*size);

		cudaMalloc( (void **) &d_sums, sizeof(double)*size);

		cudaMalloc( (void **) &d_weights, sizeof(double)*size*(inputSize+1));
		initWeights();

		cudaMalloc( (void **) &d_deltas, sizeof(double)*size);
		deltas = (double*) malloc(sizeof(double)*size);
		de_prevDeltas = t_prevLayerReference.deltaPtr;

		learnRate = t_learnRate;

		numberOfBlocks = 1;
		while(1)
		{
			numberOfThreads = size/numberOfBlocks;
			if(numberOfThreads<=800 && numberOfThreads*numberOfBlocks==size) break;
			numberOfBlocks++;

			assert(numberOfBlocks < 10 && "Could not match thread/block size");
		}

	}

	/*
	 *
	 */
	SigmoidLayer::~SigmoidLayer()
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
	void SigmoidLayer::initWeights()
	{
		double *randomValues = (double*) malloc(sizeof(double)*size*(inputSize+1));

		for(int i=0; i<(inputSize+1)*size; i++)
		{
//			std::cout << (int) (100*i/((inputSize+1)*size)) << "%\n";
			double randomValue = getRandomWeight();
			randomValues[i] = randomValue;

		}
		cudaMemcpy(d_weights, randomValues, sizeof(double)*size*(inputSize+1), cudaMemcpyHostToDevice);
		free(randomValues);
	}

	/*
	 *
	 */
	std::vector<double> SigmoidLayer::getOutput()
	{
		cudaMemcpy(output, d_output, sizeof(double)*size, cudaMemcpyDeviceToHost);

		std::vector<double> result;
		for(int i=0; i<size; i++ )
		{
			double v = output[i];
			result.push_back(v);
		}

		return result;
	}

	void SigmoidLayer::determineOutput()
	{
		determineOutputFunc<<< numberOfThreads , numberOfBlocks >>>(de_input, d_output, d_inputSize, d_sums, d_weights, d_deltas, d_b);
	}

	void SigmoidLayer::setDelta(std::vector<double> t_z)
	{
		assert(t_z.size() == size && "learning values size not match");

		#pragma omp parallel for shared(deltas,size,output, t_z) private(i) default(none)
		for(int i=0; i<size; i++ )
		{
			deltas[i] = (double) output[i] - t_z[i];
		}

		cudaMemcpy(d_deltas, deltas, sizeof(double)*size, cudaMemcpyHostToDevice);
	}

	void SigmoidLayer::learnBackPropagation()
	{
//		int64 timeBefore = cv::getTickCount();
		learnBackPropagationFunc<<< numberOfThreads , numberOfBlocks >>>(de_input, d_inputSize, d_output, d_sums, d_weights, d_deltas, de_prevDeltas, d_n, d_b);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	/*
	 *
	 */
	NeuronsPtr SigmoidLayer::getNeuronPtr()
	{
		return NeuronsPtr(d_output,size, d_deltas);
	}

	/*
	 *
	 */
//	void SigmoidLayer::saveToFile(std::ofstream & t_file)
//	{
//		t_file << (double) 1 << ' '; //Signature of SigmoidLayer
//		t_file << (double) neurons.size() << ' ';
//		t_file << learnRate << ' ';
//		t_file << b << ' ';
//
//		for( auto it = neurons.begin(); it != neurons.end(); it++)
//		{
//			std::vector<double> *weights = it->getWeights();
//			for(int i=0; i<weights->size(); i++)
//			{
//				t_file << (*weights)[i] << ' ';
//			}
//		}
//	}

	/*
	 *
	 */
//	void SigmoidLayer::loadFromFile(std::ifstream & t_file)
//	{
//		for( auto it = neurons.begin(); it != neurons.end(); it++)
//		{
//			double buff;
//			std::vector<double> *weights = it->getWeights();
//			for(int i=0; i<weights->size(); i++)
//			{
//				if(t_file.eof()) {assert("SigmoidLayer LoadFromFile: unexpected end of file");}
//				t_file >> buff;
//				(*weights)[i] = buff;
//			}
//		}
//	}
}
