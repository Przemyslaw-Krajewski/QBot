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
	__global__ void learnSGDFunc(double *t_input, int *t_inputSize,
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
	__global__ void learnAdamFunc(double *t_input, int *t_inputSize,
			double *t_output,
			double *t_sums,
			double *t_weights,
			double *t_deltas, double *t_prevDeltas,
			double *t_m,double *t_v,
			double *t_n,double *t_b,
			double *t_B1,double *t_B2)
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
		double e = exp(-(*t_b)*t_sums[index]);
		double m = 1 + e;
		double derivative = ((*t_b)*e/(m*m));
		double grad = delta*derivative; // gradient without x factor
		double grad2 = grad*grad;

		//calculate moment vectors
		t_m[weightsIndex] = (*t_B1)*t_m[weightsIndex] + (1-(*t_B1))*grad;
		t_v[weightsIndex] = (*t_B2)*t_v[weightsIndex] + (1-(*t_B2))*grad*grad;
		for(int i=0; i<inputSize; i++)
		{
			t_m[weightsIndex+i+1] = (*t_B1)*t_m[weightsIndex+i+1] + (1-(*t_B1))*grad*inputBuff[i];
			t_v[weightsIndex+i+1] = (*t_B2)*t_v[weightsIndex+i+1] + (1-(*t_B2))*grad2*inputBuff[i]*inputBuff[i];
		}

		//calculate new weights
		t_weights[weightsIndex] -= (*t_n)*t_m[weightsIndex] / (__fsqrt_rd(t_v[weightsIndex]+0.00000001));
		for(int i=0; i<inputSize; i++)
		{
//			t_weights[ weightsIndex+i+1 ] -= p*inputBuff[i];
			t_weights[weightsIndex+i+1] -= (*t_n)*t_m[weightsIndex+i+1] / (__fsqrt_rd(t_v[weightsIndex+i+1]+0.00000001));
		}

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			for(int i=0; i<*t_inputSize; i++)
			{
				int idx = weightsIndex + i + 1;
				t_prevDeltas[i] += grad * t_weights[idx] ;
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
		double b1 = 0.9, b2 = 0.999;

		size = t_size;
		de_input = t_prevLayerReference.inputPtr;

		//Parameters
		cudaMalloc( (void **) &d_n, sizeof(double));
		cudaMemcpy(d_n, &(t_learnRate), sizeof(double), cudaMemcpyHostToDevice);
		learnRate = t_learnRate;
		cudaMalloc( (void **) &d_b, sizeof(double));
		cudaMemcpy(d_b, &(t_parameterB), sizeof(double), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_B1, sizeof(double));
		cudaMemcpy(d_B1, &(b1), sizeof(double), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_B2, sizeof(double));
		cudaMemcpy(d_B2, &(b2), sizeof(double), cudaMemcpyHostToDevice);

		//Input/output
		cudaMalloc( (void **) &d_inputSize, sizeof(int));
		cudaMemcpy(d_inputSize, &(t_prevLayerReference.size), sizeof(int), cudaMemcpyHostToDevice);
		inputSize = t_prevLayerReference.size;

		cudaMalloc( (void **) &d_output, sizeof(double)*size);
		output = (double*) std::malloc(sizeof(double)*size);

		//basic to learn
		cudaMalloc( (void **) &d_sums, sizeof(double)*size);

		cudaMalloc( (void **) &d_weights, sizeof(double)*size*(inputSize+1));
		initWeights();

		cudaMalloc( (void **) &d_deltas, sizeof(double)*size);
		deltas = (double*) malloc(sizeof(double)*size);
		de_prevDeltas = t_prevLayerReference.deltaPtr;

		//additional to learn
		double *zeros = (double*) malloc(sizeof(double)*size*(inputSize+1));
		for(int i=0; i<(inputSize+1)*size; i++)	zeros[i] = 0;

		cudaMalloc( (void **) &d_m, sizeof(double)*size*(inputSize+1));
		cudaMemcpy(d_m, zeros, sizeof(double)*size*(inputSize+1), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_v, sizeof(double)*size*(inputSize+1));
		cudaMemcpy(d_v, zeros, sizeof(double)*size*(inputSize+1), cudaMemcpyHostToDevice);

		free(zeros);

		// split to blocks
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
		cudaFree(d_B1);
		cudaFree(d_B2);

		cudaFree(d_inputSize);
		cudaFree(d_output);

		cudaFree(d_sums);
		cudaFree(d_weights);
		cudaFree(d_deltas);

		cudaFree(d_m);
		cudaFree(d_v);

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

	void SigmoidLayer::learnSGD()
	{
//		int64 timeBefore = cv::getTickCount();
		learnSGDFunc<<< numberOfThreads , numberOfBlocks >>>(de_input, d_inputSize,
															 d_output,
															 d_sums,
															 d_weights,
															 d_deltas, de_prevDeltas,
															 d_n, d_b);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	void SigmoidLayer::learnAdam()
	{
//		int64 timeBefore = cv::getTickCount();
		learnAdamFunc<<< numberOfThreads , numberOfBlocks >>>(de_input, d_inputSize,
															  d_output,
															  d_sums,
															  d_weights,
															  d_deltas, de_prevDeltas,
															  d_m, d_v,
															  d_n, d_b,
															  d_B1, d_B2);
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
