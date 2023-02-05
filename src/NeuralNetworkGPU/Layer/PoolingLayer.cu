//
// Created by przemo on 27.12.2019.
//

#include "PoolingLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	__global__ void determineOutputFuncPoolAvg(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_deltas)
	{

		long indexDst = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x;
		long indexSrc = threadIdx.x+threadIdx.x +
						(threadIdx.y+threadIdx.y)*(blockDim.x+blockDim.x) +
						blockIdx.x*(blockDim.y+blockDim.y)*(blockDim.x+blockDim.x);

		float value = 0;
		int maxXSize = t_inputSize->x;
		int maxYSize = t_inputSize->y;
		value += t_input[indexSrc];
		value += maxXSize > threadIdx.x ? t_input[indexSrc+1] : 0;
		value += maxYSize > threadIdx.y ? t_input[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x] : 0;
		value += maxYSize > threadIdx.y && maxXSize > threadIdx.x ? t_input[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x+1] : 0;
		value = __fdiv_rd(value,4);

		t_output[indexDst] = value;
		t_deltas[indexDst] = 0;
	}

	/*
	 *
	 */
	__global__ void learnFuncPoolAvg(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_deltas, float *t_prevDeltas)
	{
		long indexDst = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x;
		long indexSrc = threadIdx.x+threadIdx.x +
						(threadIdx.y+threadIdx.y)*(blockDim.x+blockDim.x) +
						blockIdx.x*(blockDim.y+blockDim.y)*(blockDim.x+blockDim.x);

		float delta = __fdiv_rd(t_deltas[indexDst],4);

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			int maxXSize = t_inputSize->x;
			int maxYSize = t_inputSize->y;
			t_prevDeltas[indexSrc] = delta;
			if(maxXSize > threadIdx.x) t_prevDeltas[indexSrc+1] = delta;
			if(maxYSize > threadIdx.y) t_prevDeltas[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x] = delta;
			if(maxXSize > threadIdx.x && maxYSize > threadIdx.y) t_prevDeltas[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x+1] = delta;
		}
		//reset delta
		t_deltas[indexDst] = 0;
	}

	/*
	 *
	 */
	__global__ void determineOutputFuncPoolMax(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_deltas)
	{

		long indexDst = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x;
		long indexSrc = threadIdx.x+threadIdx.x +
						(threadIdx.y+threadIdx.y)*(blockDim.x+blockDim.x) +
						blockIdx.x*(blockDim.y+blockDim.y)*(blockDim.x+blockDim.x);

		float value = 0;
		int maxXSize = t_inputSize->x;
		int maxYSize = t_inputSize->y;
		if(maxYSize > threadIdx.y && maxXSize > threadIdx.x)
		{
			value = fmaxf(t_input[indexSrc],t_input[indexSrc+1]);
			value = fmaxf(value,t_input[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x]);
			value = fmaxf(value,t_input[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x+1]);
		}
		else if(maxXSize > threadIdx.x)
		{
			value = fmaxf(t_input[indexSrc],t_input[indexSrc+1]);
		}
		else if(maxYSize > threadIdx.y)
		{
			value = fmaxf(t_input[indexSrc],t_input[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x]);
		}
//			value = __fdiv_rd(value,4);

		t_output[indexDst] = value;
		t_deltas[indexDst] = 0;
	}

	/*
	 *
	 */
	__global__ void learnFuncPoolMax(float *t_input, TensorSize *t_inputSize,
			float *t_output,
			float *t_deltas, float *t_prevDeltas)
	{
		long indexDst = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x;
		long indexSrc = threadIdx.x+threadIdx.x +
						(threadIdx.y+threadIdx.y)*(blockDim.x+blockDim.x) +
						blockIdx.x*(blockDim.y+blockDim.y)*(blockDim.x+blockDim.x);

		float delta = __fdiv_rd(t_deltas[indexDst],4);

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			int maxXSize = t_inputSize->x;
			int maxYSize = t_inputSize->y;
			int id;
			id = indexSrc;
			if(maxXSize > threadIdx.x && t_input[indexSrc+1] > t_input[id]) id = indexSrc+1;
			if(maxYSize > threadIdx.y && t_prevDeltas[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x] > t_input[id]) id = indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x;
			if(maxXSize > threadIdx.x && maxYSize > threadIdx.y && t_input[indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x+1] > t_input[id]) id = indexSrc+(threadIdx.y+threadIdx.y)*blockDim.x+1;
			t_prevDeltas[id] = delta;
		}
		//reset delta
		t_deltas[indexDst] = 0;
	}

	/*
	 *
	 */
	PoolingLayer::PoolingLayer(NeuronsPtr t_prevLayerReference)
	{
		prevLayerId = t_prevLayerReference.id;

		size = TensorSize((t_prevLayerReference.tSize.x+1)/2,(t_prevLayerReference.tSize.y+1)/2,t_prevLayerReference.tSize.z);
		de_input = t_prevLayerReference.inputPtr;

		//Input/output
		cudaMalloc( (void **) &d_inputSize, sizeof(TensorSize));
		cudaMemcpy(d_inputSize, &(t_prevLayerReference.tSize), sizeof(TensorSize), cudaMemcpyHostToDevice);
		inputSize = t_prevLayerReference.tSize;

		cudaMalloc( (void **) &d_output, sizeof(float)*size.m);
		output = (float*) std::malloc(sizeof(float)*size.m);

		//basic to learn
		cudaMalloc( (void **) &d_deltas, sizeof(float)*size.m);
		deltas = (float*) malloc(sizeof(float)*size.m);
		de_prevDeltas = t_prevLayerReference.deltaPtr;
	}

	/*
	 *
	 */
	PoolingLayer::~PoolingLayer()
	{
		cudaFree(d_inputSize);
		cudaFree(d_output);

		cudaFree(d_deltas);

		free(output);
		free(deltas);
	}

	/*
	 *
	 */
	std::vector<double> PoolingLayer::getOutput()
	{
		cudaMemcpy(output, d_output, sizeof(float)*size.m, cudaMemcpyDeviceToHost);

		std::vector<double> result;
		for(int i=0; i<size.m; i++ )
		{
			double v = output[i];
			result.push_back(v);
		}

		return result;
	}

	void PoolingLayer::determineOutput()
	{
		dim3 threadsPerBlock(size.x, size.y);
		dim3 numBlocks(size.z);
		determineOutputFuncPoolAvg<<< threadsPerBlock , numBlocks >>>(de_input, d_inputSize,
																	    d_output,
																	    d_deltas);
	}

	void PoolingLayer::learnSGD()
	{
//		int64 timeBefore = cv::getTickCount();
		dim3 threadsPerBlock(size.x, size.y);
		dim3 numBlocks(size.z);
		learnFuncPoolAvg<<< threadsPerBlock , numBlocks >>>(de_input, d_inputSize,
															 d_output,
															 d_deltas, de_prevDeltas);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	void PoolingLayer::learnAdam()
	{
//		int64 timeBefore = cv::getTickCount();
		dim3 threadsPerBlock(size.x, size.y);
		dim3 numBlocks(size.z);
		learnFuncPoolAvg<<< threadsPerBlock , numBlocks >>>(de_input, d_inputSize,
															 d_output,
															 d_deltas, de_prevDeltas);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	/*
	 *
	 */
	NeuronsPtr PoolingLayer::getNeuronPtr()
	{
		return NeuronsPtr(layerId, d_output,size, d_deltas);
	}

	/*
	 *
	 */
	void PoolingLayer::saveToFile(std::ofstream & t_file)
	{
		t_file << (float) getLayerTypeId() << ' ';
		t_file << (float) prevLayerId << ' '; 	   //Id of previous layer
	}

	/*
	 *
	 */
	PoolingLayer* PoolingLayer::loadFromFile(std::ifstream &t_file, std::vector<NeuronsPtr> &t_prevLayerReferences)
	{
		float prevId;
		t_file >> prevId;

		return new PoolingLayer(t_prevLayerReferences[(int)prevId]);
	}
}
