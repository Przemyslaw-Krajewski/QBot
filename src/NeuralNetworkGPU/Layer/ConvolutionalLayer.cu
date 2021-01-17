//
// Created by przemo on 27.12.2019.
//

#include "ConvolutionalLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	__global__ void determineOutputFuncConv(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_sums,
			float *t_weights, MatrixSize *t_filterSize,
			float *t_deltas,
			float *d_b)
	{
//		__shared__ float inputBuff[INPUT_BUFFER_SIZE];

		//copy input to common buffer
//		if(inputSize == blockDim.x)
//		{
//			inputBuff[threadIdx.x] = t_input[threadIdx.x];
//		}
//		else if(inputSize < blockDim.x)
//		{
//			if(threadIdx.x < inputSize) inputBuff[threadIdx.x] = t_input[threadIdx.x];
//		}
//		else if(inputSize > blockDim.x)
//		{
//			int index = inputSize-threadIdx.x-1;
//			while(index >= 0)
//			{
//				inputBuff[index] = t_input[index];
//				index -= blockDim.x;
//			}
//		}
//		__syncthreads();

		//sums x[i]*w[i]
		int xHalfFilterSize = t_filterSize->x/2;
		int yHalfFilterSize = t_filterSize->y/2;
		float sum = 0;
		for(int x=-xHalfFilterSize; x<=xHalfFilterSize; x++)
		{
			for(int y=-yHalfFilterSize; y<=yHalfFilterSize; y++)
			{
				for(int z=0; z<t_inputSize->z; z++)
				{
					int fx = xHalfFilterSize+x;
					int fy = yHalfFilterSize+y;
					int tx = xHalfFilterSize+blockIdx.x+x;
					int ty = yHalfFilterSize+blockIdx.y+y;
					int tz = z;

					sum += t_input[tx + ty*t_inputSize->x + tz*t_inputSize->x*t_inputSize->y] *
								t_weights[fx + fy*t_filterSize->x + z*t_filterSize->x*t_filterSize->y];
				}
			}
		}

		t_sums[blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y] = sum;
		//activation function
		t_output[blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y] =
//				1 / (1 + exp(-(*d_b)*sum) );	//sigmoid function
				sum > 0 ? sum : sum*0.1; 					//RELU function
		//reset delta
		t_deltas[blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y] = 0;
	}

	/*
	 *
	 */
	__global__ void learnSGDConv(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_sums,
			float *t_weights, MatrixSize *t_filterSize,
			float *t_deltas, float *t_prevDeltas,
			float *d_n,float *d_b)
	{
//		int inputSize = 3;//*t_inputSize;

		//copy input to common buffer
//		__shared__ float inputBuff[INPUT_BUFFER_SIZE];
//		if(inputSize == blockDim.x)
//		{
//			inputBuff[threadIdx.x] = t_input[threadIdx.x];
//		}
//		else if(inputSize < blockDim.x)
//		{
//			if(threadIdx.x < inputSize) inputBuff[threadIdx.x] = t_input[threadIdx.x];
//		}
//		else if(inputSize > blockDim.x)
//		{
//			int index = inputSize-threadIdx.x-1;
//			while(index >= 0)
//			{
//				inputBuff[index] = t_input[index];
//				index -= blockDim.x;
//			}
//		}
//		__syncthreads();

		long index = blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y;
		float delta = t_deltas[index];

		//determine common multiplier
//		float e = exp(-(*d_b)*t_sums[index]);
//		float m = 1 + e;
//		float derivative = ((*d_b)*e/(m*m));
		float derivative = t_sums[index] > 0 ? 1 : 0.1;

		float p = (*d_n)* delta * derivative;
		//calculate new weights
		int xHalfFilterSize = t_filterSize->x/2;
		int yHalfFilterSize = t_filterSize->y/2;
		for(int x=-xHalfFilterSize; x<=xHalfFilterSize; x++)
		{
			for(int y=-yHalfFilterSize; y<=yHalfFilterSize; y++)
			{
				for(int z=0; z<t_inputSize->z; z++)
				{
					int fx = xHalfFilterSize+x;
					int fy = yHalfFilterSize+y;
					int tx = xHalfFilterSize+blockIdx.x+x;
					int ty = yHalfFilterSize+blockIdx.y+y;
					int tz = z;
					t_weights[ fx + fy*t_filterSize->x + z*t_filterSize->x*t_filterSize->y ] -=
				    		p*t_input[tx + ty*t_inputSize->x + tz*t_inputSize->x*t_inputSize->y];
				}
			}
		}

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			for(int x=-xHalfFilterSize; x<=xHalfFilterSize; x++)
			{
				for(int y=-yHalfFilterSize; y<=yHalfFilterSize; y++)
				{
					for(int z=0; z<t_inputSize->z; z++)
					{
						int fx = xHalfFilterSize+x;
						int fy = yHalfFilterSize+y;
						int tx = xHalfFilterSize+blockIdx.x+x;
						int ty = yHalfFilterSize+blockIdx.y+y;
						int tz = z;

						t_prevDeltas[tx + ty*t_inputSize->x + tz*t_inputSize->x*t_inputSize->y] +=
								delta * derivative * t_weights[ fx + fy*t_filterSize->x + z*t_filterSize->x*t_filterSize->y ];;
					}
				}
			}
		}

		//reset delta
		t_deltas[index] = 0;

	}

	/*
	 *
	 */
	ConvolutionalLayer::ConvolutionalLayer(float t_parameterB, float t_learnRate, int convLayers,
			MatrixSize t_filterSize, NeuronsPtr t_prevLayerReference)
	{
		size = TensorSize(t_prevLayerReference.tSize.x-t_filterSize.x+1,
						  t_prevLayerReference.tSize.y-t_filterSize.y+1,
						  convLayers);
		de_input = t_prevLayerReference.inputPtr;

		//learn rate
		cudaMalloc( (void **) &d_n, sizeof(float));
		cudaMemcpy(d_n, &(t_learnRate), sizeof(float), cudaMemcpyHostToDevice);
		//parameter b
		cudaMalloc( (void **) &d_b, sizeof(float));
		cudaMemcpy(d_b, &(t_parameterB), sizeof(float), cudaMemcpyHostToDevice);

		//input size
		cudaMalloc( (void **) &d_inputSize, sizeof(TensorSize));
		cudaMemcpy(d_inputSize, &t_prevLayerReference.tSize, sizeof(TensorSize), cudaMemcpyHostToDevice);
		inputSize = t_prevLayerReference.tSize;

		//output
		cudaMalloc( (void **) &d_output, sizeof(float)*size.m);
		output = (float*) std::malloc(sizeof(float)*size.m);

		//filter size
		filterSize = t_filterSize;
		cudaMalloc( (void **) &d_filterSize, sizeof(MatrixSize));
		cudaMemcpy(d_filterSize, &t_filterSize, sizeof(MatrixSize), cudaMemcpyHostToDevice);
		//weights
		cudaMalloc( (void **) &d_weights, sizeof(float)*t_filterSize.m*t_prevLayerReference.tSize.z);
		initWeights();

		//sums
		cudaMalloc( (void **) &d_sums, sizeof(float)*size.m);
		//deltas
		cudaMalloc( (void **) &d_deltas, sizeof(float)*size.m);
		deltas = (float*) malloc(sizeof(float)*size.m);
		de_prevDeltas = t_prevLayerReference.deltaPtr;

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

		cudaFree(d_filterSize);

		cudaFree(d_deltas);

		free(output);
		free(deltas);
	}

	/*
	 *
	 */
	void ConvolutionalLayer::initWeights()
	{
		long weightsSize = filterSize.m*inputSize.z;

		float *randomValues = (float*) malloc(sizeof(float)*weightsSize);

		for(int i=0; i< weightsSize; i++)
		{
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
		cudaMemcpy(output, d_output, sizeof(float)*size.m, cudaMemcpyDeviceToHost);

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
		dim3 threadsPerBlock(size.z);
		dim3 numBlocks(size.x, size.y);
		determineOutputFuncConv<<< numBlocks , threadsPerBlock >>>(de_input, d_inputSize,
																  d_output,
																  d_sums,
																  d_weights, d_filterSize,
																  d_deltas,
																  d_b);
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
		dim3 threadsPerBlock(size.z);
		dim3 numBlocks(size.x, size.y);
		learnSGDConv<<< numBlocks , threadsPerBlock >>>(de_input, d_inputSize,
														d_output,
														d_sums,
														d_weights, d_filterSize,
														d_deltas, de_prevDeltas,
														d_n, d_b);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	void ConvolutionalLayer::learnAdam()
	{
		//TODO::Implement SGD first
	}

	/*
	 *
	 */
	NeuronsPtr ConvolutionalLayer::getNeuronPtr()
	{
		return NeuronsPtr(d_output,size, d_deltas);
	}
}
