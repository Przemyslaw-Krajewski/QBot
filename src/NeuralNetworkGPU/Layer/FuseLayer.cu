//
// Created by przemo on 27.12.2019.
//

#include "FuseLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *
	 */
	__global__ void determineOutputFuncFuse(float *t_input1, int *t_inputSize1, float *t_input2, int *t_inputSize2,
			float *t_output,
			float *t_deltas)
	{
		//Just copy from 2 sources
		long index = threadIdx.x + blockIdx.x*blockDim.x;
		t_deltas[index] = 0;
		if(index < *t_inputSize1)
		{
			t_output[index] = t_input1[index];
		}
		else
		{
			t_output[index] = t_input2[index-*t_inputSize1];
		}
	}

	/*
	 *
	 */
	__global__ void learnFuncFuse(float *t_input1, int *t_inputSize1, float *t_input2, int *t_inputSize2,
			float *t_output,
			float *t_deltas, float *t_prevDeltas1, float *t_prevDeltas2)
	{
		long index = threadIdx.x +  blockIdx.x*blockDim.x;

		//reset delta
		t_deltas[index] = 0;

		//set delta to deeper neurons
		if(index < *t_inputSize1)
		{
			if(t_prevDeltas1 != nullptr) t_prevDeltas1[index] = t_deltas[index];
		}
		else
		{
			if(t_prevDeltas2 != nullptr) t_prevDeltas2[index-*t_inputSize1] = t_deltas[index];
		}
	}

	/*
	 *
	 */
	FuseLayer::FuseLayer(NeuronsPtr t_prevLayerReference1, NeuronsPtr t_prevLayerReference2)
	{
		size = t_prevLayerReference1.size + t_prevLayerReference2.size;
		de_input1 = t_prevLayerReference1.inputPtr;
		de_input2 = t_prevLayerReference2.inputPtr;

		//Input/output
		cudaMalloc( (void **) &d_inputSize1, sizeof(int));
		cudaMemcpy(d_inputSize1, &(t_prevLayerReference1.size), sizeof(int), cudaMemcpyHostToDevice);
		inputSize1 = t_prevLayerReference1.size;
		cudaMalloc( (void **) &d_inputSize2, sizeof(int));
		cudaMemcpy(d_inputSize2, &(t_prevLayerReference2.size), sizeof(int), cudaMemcpyHostToDevice);
		inputSize2 = t_prevLayerReference2.size;

		cudaMalloc( (void **) &d_output, sizeof(float)*size);
		output = (float*) std::malloc(sizeof(float)*size);

		//basic to learn
		cudaMalloc( (void **) &d_deltas, sizeof(float)*size);
		deltas = (float*) malloc(sizeof(float)*size);
		de_prevDeltas1 = t_prevLayerReference1.deltaPtr;
		de_prevDeltas2 = t_prevLayerReference2.deltaPtr;

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
	FuseLayer::~FuseLayer()
	{
		cudaFree(d_inputSize1);
		cudaFree(d_inputSize2);
		cudaFree(d_output);

		cudaFree(d_deltas);

		free(output);
		free(deltas);
	}

	/*
	 *
	 */
	std::vector<double> FuseLayer::getOutput()
	{
		cudaMemcpy(output, d_output, sizeof(float)*size, cudaMemcpyDeviceToHost);

		std::vector<double> result;
		for(int i=0; i<size; i++ )
		{
			double v = output[i];
			result.push_back(v);
		}

		return result;
	}

	void FuseLayer::determineOutput()
	{
		determineOutputFuncFuse<<< numberOfThreads , numberOfBlocks >>>(de_input1, d_inputSize1, de_input2, d_inputSize2,
																	d_output,
																	d_deltas);
	}

	void FuseLayer::setDelta(std::vector<double> t_z)
	{
		assert(t_z.size() == size && "learning values size not match");

		#pragma omp parallel for shared(deltas,size,output, t_z) private(i) default(none)
		for(int i=0; i<size; i++ )
		{
			deltas[i] = (float) output[i] - t_z[i];
		}

		cudaMemcpy(d_deltas, deltas, sizeof(float)*size, cudaMemcpyHostToDevice);
	}

	void FuseLayer::learnSGD()
	{
//		int64 timeBefore = cv::getTickCount();

		learnFuncFuse<<< numberOfThreads , numberOfBlocks >>>(de_input1, d_inputSize1, de_input2, d_inputSize2,
															 d_output,
															 d_deltas, de_prevDeltas1, de_prevDeltas2);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	void FuseLayer::learnAdam()
	{
//		int64 timeBefore = cv::getTickCount();
		learnFuncFuse<<< numberOfThreads , numberOfBlocks >>>(de_input1, d_inputSize1, de_input2, d_inputSize2,
																	 d_output,
																	 d_deltas, de_prevDeltas1, de_prevDeltas2);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	/*
	 *
	 */
	NeuronsPtr FuseLayer::getNeuronPtr()
	{
		return NeuronsPtr(d_output,size, d_deltas);
	}
}
