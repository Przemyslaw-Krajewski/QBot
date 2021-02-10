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
		long index = blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y;

		//sums x[i]*w[i]
		int yFrame = t_inputSize->x*t_inputSize->y;
		int yfFrame = t_filterSize->y*t_filterSize->x;
		int yOffset = blockIdx.y*t_inputSize->x;
		int zfOffset = yfFrame*t_inputSize->z*threadIdx.x;
		float sum = 0;

		for(int y=0,yf=0,yi=yOffset; y<t_filterSize->y; y++)
		{
			for(int x=0; x<t_filterSize->x; x++)
			{
				for(int z=0,zf=zfOffset,zi=0; z<t_inputSize->z; z++)
				{
					sum += t_input[blockIdx.x+x + yi + zi] * t_weights[x + yf + zf];
					zf+=yfFrame;
					zi+=yFrame;
				}
			}
			yf+=t_filterSize->x;
			yi+=t_inputSize->y;
		}

		t_sums[index] = sum;
		//reset delta
		t_deltas[index] = 0;
		//activation function
		t_output[index] =
//				1 / (1 + exp(-(*d_b)*sum) );	//sigmoid function
				sum > 0 ? sum : sum*0.1; 					//RELU function
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
		long index = blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y;
		float delta = t_deltas[index];

		//determine common multiplier
		float e = exp(-(*d_b)*t_sums[index]);
		float m = 1 + e;
		float derivative = ((*d_b)*e/(m*m));
//		float derivative = t_sums[index] > 0 ? 1 : 0.1;

		float p = (*d_n)* delta * derivative;
		//calculate new weights
		int yFrame = t_inputSize->x*t_inputSize->y;
		int yfFrame = t_filterSize->y*t_filterSize->x;
		int yOffset = blockIdx.y*t_inputSize->x;
		int zfOffset = yfFrame*t_inputSize->z*threadIdx.x;
		for(int y=0,yf=0,yi=yOffset; y<t_filterSize->y; y++)
		{
			for(int x=0; x<t_filterSize->x; x++)
			{
				for(int z=0,zf=zfOffset,zi=0; z<t_inputSize->z; z++)
				{
					t_weights[ x + yf + zf ] -= p*t_input[blockIdx.x+x + yi + zi];

					zf+=yfFrame;
					zi+=yFrame;
				}
			}
			yf+=t_filterSize->x;
			yi+=t_inputSize->y;
		}

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			float dd = delta*derivative;
			for(int y=0,yf=0,yi=yOffset; y<t_filterSize->y; y++)
			{
				for(int x=0; x<t_filterSize->x; x++)
				{
					for(int z=0,zf=0,zi=0; z<t_inputSize->z; z++)
					{
						t_prevDeltas[blockIdx.x+x + yi + zi] += dd * t_weights[ x + yf + zf ];

						zf+=yfFrame;
						zi+=yFrame;
					}
				}
				yf+=t_filterSize->x;
				yi+=t_inputSize->y;
			}
		}

		//reset delta
		t_deltas[index] = 0;

	}

	/*
	 *
	 */
	__global__ void learnAdamConv(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_sums,
			float *t_weights, MatrixSize *t_filterSize,
			float *t_deltas, float *t_prevDeltas,
			float *t_m,float *t_v,
			float *t_n,float *t_b,
			float *t_B1,float *t_B2)
	{
		long index = blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y;
		float delta = t_deltas[index];

		//determine derivative and gradients
		float e = exp(-(*t_b)*t_sums[index]);
		float m = 1 + e;
		float derivative = ((*t_b)*e/(m*m));
//		float sum = t_sums[index];
//		float derivative = sum > 0 && sum < 4080 ? 1 : 0.1;
		float grad = delta*derivative; // gradient without x factor
		float grad2 = grad*grad;

		//calculate new weights
		int yFrame = t_inputSize->x*t_inputSize->y;
		int yfFrame = t_filterSize->y*t_filterSize->x;
		int yOffset = blockIdx.y*t_inputSize->x;
		int zfOffset = yfFrame*t_inputSize->z*threadIdx.x;
		float mTarget,vTarget;
		float mNew, vNew;
		for(int y=0,yf=0,yi=yOffset; y<t_filterSize->y; y++)
		{
			for(int x=0; x<t_filterSize->x; x++)
			{
				for(int z=0,zf=zfOffset,zi=0; z<t_inputSize->z; z++)
				{
					float input = t_input[blockIdx.x+x + yi + zi];
					//calculate new m & v
					mTarget = grad*input;
					vTarget = grad2*input*input;
					mNew = mTarget - (*t_B1)*(mTarget-t_m[x + yf + zf]);
					vNew = vTarget - (*t_B2)*(vTarget-t_v[x + yf + zf]);
					t_m[x + yf + zf] = mNew;
					t_v[x + yf + zf] = vNew;

					//update weights
					t_weights[x + yf + zf] -= __fdiv_rd ((*t_n)*mNew , (__fsqrt_rd(vNew)+0.00001));

					zf+=yfFrame;
					zi+=yFrame;
				}
			}
			yf+=t_filterSize->x;
			yi+=t_inputSize->y;
		}

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			float dd = delta*derivative;
			for(int y=0,yf=0,yi=yOffset; y<t_filterSize->y; y++)
			{
				for(int x=0; x<t_filterSize->x; x++)
				{
					for(int z=0,zf=0,zi=0; z<t_inputSize->z; z++)
					{
						t_prevDeltas[blockIdx.x+x + yi + zi] += dd * t_weights[ x + yf + zf ];

						zf+=yfFrame;
						zi+=yFrame;
					}
				}
				yf+=t_filterSize->x;
				yi+=t_inputSize->y;
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
		float b1 = 0.9, b2 = 0.999;

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
		//Adam parameters
		cudaMalloc( (void **) &d_B1, sizeof(float));
		cudaMemcpy(d_B1, &(b1), sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_B2, sizeof(float));
		cudaMemcpy(d_B2, &(b2), sizeof(float), cudaMemcpyHostToDevice);

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
		int weightsSize = t_filterSize.m*t_prevLayerReference.tSize.z*convLayers;
		cudaMalloc( (void **) &d_weights, sizeof(float)*weightsSize);
		initWeights();

		//sums
		cudaMalloc( (void **) &d_sums, sizeof(float)*size.m);
		//deltas
		cudaMalloc( (void **) &d_deltas, sizeof(float)*size.m);
		deltas = (float*) malloc(sizeof(float)*size.m);
		de_prevDeltas = t_prevLayerReference.deltaPtr;

		//adam learn
		float *zeros = (float*) malloc(sizeof(float)*weightsSize);
		for(int i=0; i<weightsSize; i++)	zeros[i] = 0;

		cudaMalloc( (void **) &d_m, sizeof(float)*weightsSize);
		cudaMemcpy(d_m, zeros, sizeof(float)*weightsSize, cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_v, sizeof(float)*weightsSize);
		cudaMemcpy(d_v, zeros, sizeof(float)*weightsSize, cudaMemcpyHostToDevice);

		free(zeros);

	}

	/*
	 *
	 */
	ConvolutionalLayer::~ConvolutionalLayer()
	{
		cudaFree(d_n);
		cudaFree(d_b);
		cudaFree(d_B1);
		cudaFree(d_B2);

		cudaFree(d_inputSize);
		cudaFree(d_output);
		cudaFree(d_sums);
		cudaFree(d_weights);

		cudaFree(d_filterSize);

		cudaFree(d_deltas);

		cudaFree(d_m);
		cudaFree(d_v);

		free(output);
		free(deltas);
	}

	/*
	 *
	 */
	void ConvolutionalLayer::initWeights()
	{
		long weightsSize = filterSize.m*inputSize.z*size.z;

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
		dim3 threadsPerBlock(size.z);
		dim3 numBlocks(size.x, size.y);
		learnAdamConv<<< numBlocks , threadsPerBlock >>>(de_input, d_inputSize,
														d_output,
														d_sums,
														d_weights, d_filterSize,
														d_deltas, de_prevDeltas,
														d_m, d_v,
														d_n, d_b,
														d_B1, d_B2);
	}

	/*
	 *
	 */
	NeuronsPtr ConvolutionalLayer::getNeuronPtr()
	{
		return NeuronsPtr(d_output,size, d_deltas);
	}
}
