//
// Created by przemo on 27.12.2019.
//

#include "ConvSeparateWeightsLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	__global__ void determineOutputFuncConvSW(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_sums,
			float *t_weights, MatrixSize *t_filterSize,
			float *t_deltas,
			float *d_b,
			int   *d_stride)
	{
		long index = blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y;

		//sums x[i]*w[i]
		int yFrame = t_inputSize->x*t_inputSize->y;
		int yfFrame = t_filterSize->y*t_filterSize->x;
		int xOffset = *d_stride*blockIdx.x;
		int yOffset = *d_stride*blockIdx.y*t_inputSize->x;
		int zfOffset = yfFrame*t_inputSize->z*index;
		float sum = 0;

		for(int y=0,yf=0,yi=yOffset; y<t_filterSize->y; y++)
		{
			for(int x=0; x<t_filterSize->x; x++)
			{
				for(int z=0,zf=zfOffset,zi=0; z<t_inputSize->z; z++)
				{
					sum += t_input[xOffset+x + yi + zi] * t_weights[x + yf + zf];
					zf+=yfFrame;
					zi+=yFrame;
				}
			}
			yf+=t_filterSize->x;
			yi+=t_inputSize->y;
		}

		t_sums[index] = sum;
//		sum = sum > 255 ? 255 : sum;
//		sum = sum < -255 ? -255 : sum;
		//reset delta
		t_deltas[index] = 0;
		//activation function
		t_output[index] =
				1 / (1 + exp(-(*d_b)*sum) );	//sigmoid function
//				sum > 0 ? sum : sum*0.05; 		//RELU function
	}

	/*
	 *
	 */
	__global__ void learnSGDConvSW(float *t_input, TensorSize *t_inputSize,
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
//		float derivative = t_sums[index] > 0 ? 1 : 0.05;

		float p = (*d_n)* delta * derivative;
		//calculate new weights
		int yFrame = t_inputSize->x*t_inputSize->y;
		int yfFrame = t_filterSize->y*t_filterSize->x;
		int yOffset = blockIdx.y*t_inputSize->x;
		int zfOffset = yfFrame*t_inputSize->z*index;
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
	__global__ void learnAdamConvSW(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_sums,
			float *t_weights, MatrixSize *t_filterSize,
			float *t_deltas, float *t_prevDeltas,
			float *t_m,float *t_v,
			float *t_n,float *t_b,
			float *t_B1,float *t_B2,
			int   *d_stride)
	{
		long index = blockIdx.x + blockIdx.y*gridDim.x + threadIdx.x*gridDim.x*gridDim.y;
		float delta = t_deltas[index];

		//determine derivative and gradients
		float e = exp(-(*t_b)*t_sums[index]);
		float m = 1 + e;
		float derivative = ((*t_b)*e/(m*m));
//		float sum = t_sums[index];
//		float derivative = sum > 0 && sum < 65536 ? 1 : 0.05;
		float grad = delta*derivative; // gradient without x factor
		float grad2 = grad*grad;

		//calculate new weights
		int yFrame = t_inputSize->x*t_inputSize->y;
		int yfFrame = t_filterSize->y*t_filterSize->x;
		int xOffset = *d_stride*blockIdx.x;
		int yOffset = *d_stride*blockIdx.y*t_inputSize->x;
		int zfOffset = yfFrame*t_inputSize->z*index;
		float mTarget,vTarget;
		float mNew, vNew;
		for(int y=0,yf=0,yi=yOffset; y<t_filterSize->y; y++)
		{
			for(int x=0; x<t_filterSize->x; x++)
			{
				for(int z=0,zf=zfOffset,zi=0; z<t_inputSize->z; z++)
				{
					float input = t_input[xOffset+x + yi + zi];
					//calculate new m & v
					mTarget = grad*input;
					vTarget = grad2*input*input;
					mNew = mTarget - (*t_B1)*(mTarget-t_m[x + yf + zf]);
					vNew = vTarget - (*t_B2)*(vTarget-t_v[x + yf + zf]);
					t_m[x + yf + zf] = mNew;
					t_v[x + yf + zf] = vNew;

					//update weights
					t_weights[x + yf + zf] -= __fdiv_rd ((*t_n)*mNew , (__fsqrt_rd(vNew)+0.0000001));

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
					for(int z=0,zf=zfOffset,zi=0; z<t_inputSize->z; z++)
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
	__global__ void scaleWeightsConvSW(TensorSize *t_inputSize,
			float *t_weights, MatrixSize *t_filterSize)
	{
		//calculate new weights
		int yfFrame = t_filterSize->y*t_filterSize->x;
		int zfOffset = yfFrame*t_inputSize->z*threadIdx.x;
		float sum = 0;
		for(int y=0,yf=0; y<t_filterSize->y; y++)
		{
			for(int x=0; x<t_filterSize->x; x++)
			{
				for(int z=0,zf=zfOffset; z<t_inputSize->z; z++)
				{
					sum = abs(t_weights[ x + yf + zf ]) > sum ? abs(t_weights[ x + yf + zf ]) : sum;

					zf+=yfFrame;
				}
			}
			yf+=t_filterSize->x;
		}
		__fdiv_rd(sum,100);
		for(int y=0,yf=0; y<t_filterSize->y; y++)
		{
			for(int x=0; x<t_filterSize->x; x++)
			{
				for(int z=0,zf=zfOffset; z<t_inputSize->z; z++)
				{
					t_weights[ x + yf + zf ] = __fdiv_rd(t_weights[ x + yf + zf ],sum);

					zf+=yfFrame;
				}
			}
			yf+=t_filterSize->x;
		}
	}

	/*
	 *
	 */
	ConvSeparateWeightsLayer::ConvSeparateWeightsLayer(float t_parameterB, float t_learnRate, int convLayers,
			MatrixSize t_filterSize, NeuronsPtr t_prevLayerReference, int t_stride)
	{
		float b1 = 0.9, b2 = 0.999;

		prevLayerId = t_prevLayerReference.id;

		size = TensorSize((t_prevLayerReference.tSize.x-t_filterSize.x+1)/t_stride,
						  (t_prevLayerReference.tSize.y-t_filterSize.y+1)/t_stride,
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
		//stride
		int stride = t_stride;
		cudaMalloc( (void **) &d_stride, sizeof(int));
		cudaMemcpy(d_stride, &(stride), sizeof(int), cudaMemcpyHostToDevice);

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
		long weightsSize = t_filterSize.m*t_prevLayerReference.tSize.z*size.z*size.y*size.x;
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
	ConvSeparateWeightsLayer::~ConvSeparateWeightsLayer()
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
	void ConvSeparateWeightsLayer::initWeights()
	{
		long weightsSize = filterSize.m*inputSize.z*size.z*size.y*size.x;

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
	void ConvSeparateWeightsLayer::setWeights(float* t_weights)
	{
		long weightsSize = filterSize.m*inputSize.z*size.z*size.y*size.x;
		cudaMemcpy(d_weights, t_weights, sizeof(float)*weightsSize, cudaMemcpyHostToDevice);
	}

	/*
	 *
	 */
	void ConvSeparateWeightsLayer::setMomentum1(float* t_momentum)
	{
		long weightsSize = filterSize.m*inputSize.z*size.z*size.y*size.x;
		cudaMemcpy(d_m, t_momentum, sizeof(float)*weightsSize, cudaMemcpyHostToDevice);
	}

	/*
	 *
	 */
	void ConvSeparateWeightsLayer::setMomentum2(float* t_momentum)
	{
		long weightsSize = filterSize.m*inputSize.z*size.z*size.y*size.x;
		cudaMemcpy(d_v, t_momentum, sizeof(float)*weightsSize, cudaMemcpyHostToDevice);
	}

	/*
	 *
	 */
	std::vector<double> ConvSeparateWeightsLayer::getOutput()
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

	void ConvSeparateWeightsLayer::determineOutput()
	{
		dim3 threadsPerBlock(size.z);
		dim3 numBlocks(size.x, size.y);
		determineOutputFuncConvSW<<< numBlocks , threadsPerBlock >>>(de_input, d_inputSize,
																  d_output,
																  d_sums,
																  d_weights, d_filterSize,
																  d_deltas,
																  d_b,
																  d_stride);
	}

	void ConvSeparateWeightsLayer::learnSGD()
	{
//		int64 timeBefore = cv::getTickCount();
		dim3 threadsPerBlock(size.z);
		dim3 numBlocks(size.x, size.y);
		learnSGDConvSW<<< numBlocks , threadsPerBlock >>>(de_input, d_inputSize,
														d_output,
														d_sums,
														d_weights, d_filterSize,
														d_deltas, de_prevDeltas,
														d_n, d_b);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	void ConvSeparateWeightsLayer::learnAdam()
	{
		dim3 threadsPerBlock(size.z);
		dim3 numBlocks(size.x, size.y);
		learnAdamConvSW<<< numBlocks , threadsPerBlock >>>(de_input, d_inputSize,
														d_output,
														d_sums,
														d_weights, d_filterSize,
														d_deltas, de_prevDeltas,
														d_m, d_v,
														d_n, d_b,
														d_B1, d_B2,
														d_stride);
	}

	/*
	 *
	 */
	NeuronsPtr ConvSeparateWeightsLayer::getNeuronPtr()
	{
		return NeuronsPtr(layerId, d_output,size, d_deltas);
	}

	/*
	 *
	 */
	void ConvSeparateWeightsLayer::saveToFile(std::ofstream &t_file)
	{
		t_file << (float) getLayerTypeId() << ' '; //Signature of SigmoidLayer
		t_file << (float) prevLayerId << ' '; 	   //Id of previous layer

		t_file << (float) inputSize.x << ' ';
		t_file << (float) inputSize.y << ' ';
		t_file << (float) inputSize.z << ' ';

		t_file << (float) size.z << ' ';

		t_file << (float) filterSize.x << ' ';
		t_file << (float) filterSize.y << ' ';

		float learnRate;
		cudaMemcpy(&learnRate, d_n, sizeof(float), cudaMemcpyDeviceToHost);
		t_file << learnRate << ' ';
		float b;
		cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
		t_file << b << ' ';
		int stride;
		cudaMemcpy(&stride, d_stride, sizeof(int), cudaMemcpyDeviceToHost);
		t_file << (float) stride << ' ';

		//Weights
		long weightsSize = filterSize.m*inputSize.z*size.z*size.y*size.x;
		float *weights = (float*) malloc(sizeof(float)*weightsSize);

		cudaMemcpy(weights, d_weights, sizeof(float)*weightsSize, cudaMemcpyDeviceToHost);
		for(int i=0; i< weightsSize; i++)
		{
			t_file << weights[i] << ' ';
		}

		cudaMemcpy(weights, d_m, sizeof(float)*weightsSize, cudaMemcpyDeviceToHost);
		for(int i=0; i< weightsSize; i++)
		{
			t_file << weights[i] << ' ';
		}

		cudaMemcpy(weights, d_v, sizeof(float)*weightsSize, cudaMemcpyDeviceToHost);
		for(int i=0; i< weightsSize; i++)
		{
			t_file << weights[i] << ' ';
		}

		free(weights);
	}

	/*
	 *
	 */

	ConvSeparateWeightsLayer* ConvSeparateWeightsLayer::loadFromFile(std::ifstream &t_file, std::vector<NeuronsPtr> &t_prevLayerReferences)
	{
		float filterSize[2], convSize, inputSize[3];
		float learnRate, b, stridef;
		float prevId;
		int stride;

		t_file >> prevId;

		t_file >> inputSize[0];
		t_file >> inputSize[1];
		t_file >> inputSize[2];

		t_file >> convSize;

		t_file >> filterSize[0];
		t_file >> filterSize[1];

		t_file >> learnRate;
		t_file >> b;
		t_file >> stridef;
		stride = stridef;

		ConvSeparateWeightsLayer* layer = new ConvSeparateWeightsLayer(b,
														   	   	   	   learnRate,
																	   convSize,
																	   MatrixSize(filterSize[0],filterSize[1]),
																	   t_prevLayerReferences[(int)prevId],
																	   stride);

		long weightsSize = filterSize[0]*filterSize[1]*inputSize[2]*convSize
								*((int)((inputSize[0]-filterSize[0]+1)/stride))	//size.x
								*((int)((inputSize[1]-filterSize[1]+1)/stride));	//size.y
		float *weights = (float*) malloc(sizeof(float)*weightsSize);
		float buff;
		for(int i=0; i<weightsSize; i++)
		{
			t_file >> buff;
			weights[i] = buff;
		}
		layer->setWeights(weights);

		for(int i=0; i<weightsSize; i++)
		{
			t_file >> buff;
			weights[i] = buff;
		}
		layer->setMomentum1(weights);

		for(int i=0; i<weightsSize; i++)
		{
			t_file >> buff;
			weights[i] = buff;
		}
		layer->setMomentum2(weights);

		free(weights);

		return layer;
	}

	/*
	 *
	 */
	void ConvSeparateWeightsLayer::drawLayer()
	{
		std::vector<double> output = getOutput();
		for(int z=0; z<size.z; z++)
		{
			cv::Mat image = cv::Mat(size.y, size.x, CV_8UC3);
			for(int y=0; y<size.y; y++)
			{
				for(int x=0; x<size.x; x++)
				{
					uchar* ptrDst = image.ptr(y)+(x+x+x);
					int src = output[z*size.x*size.y + y*size.x + x]*255;
					ptrDst[0] = src;
					ptrDst[1] = src;
					ptrDst[2] = src;
				}
			}
			cv::resize(image, image, cv::Size(), 8, 8,cv::INTER_CUBIC);
			//Print
			imshow(std::to_string(z), image);
			cv::waitKey(3);
		}
	}

	/*
	 *
	 */
	void ConvSeparateWeightsLayer::printInfo()
	{
		TensorSize prevTSize;
		cudaMemcpy(&prevTSize, d_inputSize, sizeof(TensorSize), cudaMemcpyDeviceToHost);
		int weightsSize = filterSize.m*prevTSize.z*size.z*size.y*size.x;

		std::cout << "	(" << layerId << ") ConvSep <-- " << prevLayerId << " : ";
		std::cout << prevTSize.x << "x" << prevTSize.y << "x" << prevTSize.z << " -> " << size.x << "x" << size.y << "x" << size.z;
		std::cout << "   w:" << weightsSize << "\n";
	}
}
