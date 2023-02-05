//
// Created by przemo on 27.12.2019.
//

#include "SigmoidLayer.h"

#include "ActivationFunctions.h"

namespace NeuralNetworkGPU
{
	/*
	 *
	 */
	template<ActivationFunction F>
	__global__
	void determineOutputFunc(float *t_input, float *t_output, int *t_inputSize,
			float *t_sums,
			float *t_weights,
			float *t_deltas,
			float *d_b)
	{
		int inputSize = (*t_inputSize);
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

		//sums x[i]*w[i]
		long weightsIndex = inputSize*(threadIdx.x + blockIdx.x*blockDim.x);
		float sum = t_weights[weightsIndex];
		for(int i=0; i<inputSize;i++)
		{
			sum += inputBuff[i] * t_weights[ weightsIndex+i+1 ];
		}
		t_sums[threadIdx.x + blockIdx.x*blockDim.x] = sum;
		//activation function
		t_output[threadIdx.x + blockIdx.x*blockDim.x] = activationFunctionKernel<F>(&sum, d_b);
		//reset delta
		t_deltas[threadIdx.x + blockIdx.x*blockDim.x] = 0;
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	__global__
	void learnSGDFunc(float *t_input, int *t_inputSize,
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
		float derivative = derivativeFunctionKernel<F>(&t_sums[index],d_b);

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
	template<ActivationFunction F>
	__global__
	void learnAdamFunc(float *t_input, int *t_inputSize,
			float *t_output,
			float *t_sums,
			float *t_weights,
			float *t_deltas, float *t_prevDeltas,
			float *t_m,float *t_v,
			float *t_n,float *t_b,
			float *t_B1,float *t_B2)
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

		long weightsIndex = inputSize*(index);
		//determine common multiplier
		float derivative = derivativeFunctionKernel<F>(&t_sums[index],t_b);
		float grad = t_deltas[index]*derivative; // gradient without x factor
		float grad2 = grad*grad;

		//calculate new moment vectors and weights
		float mNew,vNew;
		mNew = grad - (*t_B1)*(grad-t_m[weightsIndex]);
		vNew = grad2 - (*t_B2)*(grad2-t_v[weightsIndex]);
		t_weights[weightsIndex] -= __fdiv_rd( (*t_n)*mNew , (__fsqrt_rd(vNew)+0.00001));
		t_m[weightsIndex] = mNew;
		t_v[weightsIndex] = vNew;

		float mTarget,vTarget;
		for(int i=0, indx=weightsIndex+1; i<inputSize; i++,indx++)
		{
			mTarget = grad*inputBuff[i];
			vTarget = grad2*inputBuff[i]*inputBuff[i];
			mNew = mTarget - (*t_B1)*(mTarget-t_m[indx]);
			vNew = vTarget - (*t_B2)*(vTarget-t_v[indx]);
			t_weights[indx] -= __fdiv_rd ((*t_n)*mNew , (__fsqrt_rd(vNew)+0.00001));
			t_m[indx] = mNew;
			t_v[indx] = vNew;
		}

		//set delta to deeper neurons
		if(t_prevDeltas != nullptr)
		{
			for(int i=0; i<*t_inputSize; i++)
			{
				t_prevDeltas[i] += grad * t_weights[weightsIndex+i+1] ;
			}
		}

		//reset delta
		t_deltas[index] = 0;

	}


	template class NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Sigmoid>;
	template class NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::Linear>;
	template class NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::RELU>;
	template class NeuralNetworkGPU::SigmoidLayer<NeuralNetworkGPU::ActivationFunction::LeakRELU>;

	/*
	 *
	 */
	template<ActivationFunction F>
	SigmoidLayer<F>::SigmoidLayer(float t_parameterB, float t_learnRate, int t_size, NeuronsPtr t_prevLayerReference)
	{
		float b1 = 0.9, b2 = 0.999;

		prevLayerId = t_prevLayerReference.id;

		size = t_size;
		de_input = t_prevLayerReference.inputPtr;

		//Parameters
		cudaMalloc( (void **) &d_n, sizeof(float));
		cudaMemcpy(d_n, &(t_learnRate), sizeof(float), cudaMemcpyHostToDevice);
		learnRate = t_learnRate;
		cudaMalloc( (void **) &d_b, sizeof(float));
		cudaMemcpy(d_b, &(t_parameterB), sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_B1, sizeof(float));
		cudaMemcpy(d_B1, &(b1), sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_B2, sizeof(float));
		cudaMemcpy(d_B2, &(b2), sizeof(float), cudaMemcpyHostToDevice);

		//Input/output
		cudaMalloc( (void **) &d_inputSize, sizeof(int));
		cudaMemcpy(d_inputSize, &(t_prevLayerReference.size), sizeof(int), cudaMemcpyHostToDevice);
		inputSize = t_prevLayerReference.size;

		cudaMalloc( (void **) &d_output, sizeof(float)*size);
		output = (float*) std::malloc(sizeof(float)*size);

		//basic to learn
		cudaMalloc( (void **) &d_sums, sizeof(float)*size);

		cudaMalloc( (void **) &d_weights, sizeof(float)*size*(inputSize+1));
		initWeights();

		cudaMalloc( (void **) &d_deltas, sizeof(float)*size);
		deltas = (float*) malloc(sizeof(float)*size);
		de_prevDeltas = t_prevLayerReference.deltaPtr;

		//additional to learn
		float *zeros = (float*) malloc(sizeof(float)*size*(inputSize+1));
		for(int i=0; i<(inputSize+1)*size; i++)	zeros[i] = 0;

		cudaMalloc( (void **) &d_m, sizeof(float)*size*(inputSize+1));
		cudaMemcpy(d_m, zeros, sizeof(float)*size*(inputSize+1), cudaMemcpyHostToDevice);
		cudaMalloc( (void **) &d_v, sizeof(float)*size*(inputSize+1));
		cudaMemcpy(d_v, zeros, sizeof(float)*size*(inputSize+1), cudaMemcpyHostToDevice);

		free(zeros);

		// split to blocks
		numberOfBlocks = 1;
		while(1)
		{
			numberOfThreads = size/numberOfBlocks;
			if(numberOfThreads<=800 && numberOfThreads*numberOfBlocks==size) break;
			numberOfBlocks++;

			assert(numberOfBlocks < 20 && "Could not match thread/block size");
		}
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	SigmoidLayer<F>::~SigmoidLayer()
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
	template<ActivationFunction F>
	void SigmoidLayer<F>::initWeights()
	{
		float *randomValues = (float*) malloc(sizeof(float)*size*(inputSize+1));

		for(int i=0; i<(inputSize+1)*size; i++)
		{
			float randomValue = getRandomWeight();
			randomValues[i] = randomValue;

		}
		cudaMemcpy(d_weights, randomValues, sizeof(float)*size*(inputSize+1), cudaMemcpyHostToDevice);
		free(randomValues);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::setWeights(float* t_weights)
	{
		cudaMemcpy(d_weights, t_weights, sizeof(float)*size*(inputSize+1), cudaMemcpyHostToDevice);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::setMomentum1(float* t_momentum)
	{
		cudaMemcpy(d_m, t_momentum, sizeof(float)*size*(inputSize+1), cudaMemcpyHostToDevice);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::setMomentum2(float* t_momentum)
	{
		cudaMemcpy(d_v, t_momentum, sizeof(float)*size*(inputSize+1), cudaMemcpyHostToDevice);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	std::vector<double> SigmoidLayer<F>::getOutput()
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

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::determineOutput()
	{
		determineOutputFunc<F><<< numberOfThreads , numberOfBlocks >>>(de_input, d_output, d_inputSize, d_sums, d_weights, d_deltas, d_b);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::setDelta(std::vector<double> t_z)
	{
//		#pragma omp parallel for shared(deltas,size,output, t_z) private(i) default(none)
		for(int i=0; i<size; i++ )
		{
			deltas[i] = (float) t_z[i];
		}

		cudaMemcpy(d_deltas, deltas, sizeof(float)*size, cudaMemcpyHostToDevice);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::learnSGD()
	{
//		int64 timeBefore = cv::getTickCount();
		learnSGDFunc<F><<< numberOfThreads , numberOfBlocks >>>(de_input, d_inputSize,
																						  d_output,
																						  d_sums,
																						  d_weights,
																						  d_deltas, de_prevDeltas,
																						  d_n, d_b);
//		int64 afterBefore = cv::getTickCount();
//		std::cout << "Sigm: " << (afterBefore - timeBefore)/ cv::getTickFrequency() << "\n";
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::learnAdam()
	{
//		int64 timeBefore = cv::getTickCount();
		learnAdamFunc<F><<< numberOfThreads , numberOfBlocks >>>(de_input,
																						   d_inputSize,
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
	template<ActivationFunction F>
	NeuronsPtr SigmoidLayer<F>::getNeuronPtr()
	{
		return NeuronsPtr(layerId, d_output,size, d_deltas);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::saveToFile(std::ofstream & t_file)
	{
		t_file << (float) getLayerTypeId() << ' '; //Signature of SigmoidLayer
		t_file << (float) prevLayerId << ' '; 	   //Id of previous layer
		t_file << (float) size << ' ';
		t_file << (float) inputSize << ' ';
		t_file << (float) learnRate << ' ';
		float b;
		cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
		t_file << b << ' ';

		float *weights = (float*) malloc(sizeof(float)*size*(inputSize+1));

		cudaMemcpy(weights, d_weights, sizeof(float)*size*(inputSize+1), cudaMemcpyDeviceToHost);
		for(int i=0; i<(inputSize+1)*size; i++)
		{
			t_file << weights[i] << ' ';
		}

		cudaMemcpy(weights, d_m, sizeof(float)*size*(inputSize+1), cudaMemcpyDeviceToHost);
		for(int i=0; i<(inputSize+1)*size; i++)
		{
			t_file << weights[i] << ' ';
		}

		cudaMemcpy(weights, d_v, sizeof(float)*size*(inputSize+1), cudaMemcpyDeviceToHost);
		for(int i=0; i<(inputSize+1)*size; i++)
		{
			t_file << weights[i] << ' ';
		}

		free(weights);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	SigmoidLayer<F>* SigmoidLayer<F>::loadFromFile(std::ifstream & t_file, std::vector<NeuronsPtr> &t_prevLayerReferences)
	{
		float size, inputSize, learnRate, b;
		float prevId;
		t_file >> prevId;
		t_file >> size;
		t_file >> inputSize;
		t_file >> learnRate;
		t_file >> b;

		SigmoidLayer<F>* layer = new SigmoidLayer<F>(b,learnRate,size,t_prevLayerReferences[(int)prevId]);

		float *weights = (float*) malloc(sizeof(float)*size*(inputSize+1));
		float buff;
		for(int i=0; i<(inputSize+1)*size; i++)
		{
			t_file >> buff;
			weights[i] = buff;
		}
		layer->setWeights(weights);

		for(int i=0; i<(inputSize+1)*size; i++)
		{
			t_file >> buff;
			weights[i] = buff;
		}
		layer->setMomentum1(weights);

		for(int i=0; i<(inputSize+1)*size; i++)
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
	template<ActivationFunction F>
	void SigmoidLayer<F>::drawLayer()
	{
		int blockSize=20;
		int spaceSize=2;
		std::vector<double> output = getOutput();

		cv::Mat mat = cv::Mat(1500, 1500, CV_8UC3);

		for(int x=0; x<mat.cols; x++)
		{
			for(int y=0; y<mat.rows; y++)
			{
				uchar* ptr = mat.ptr(y)+(x)*3;
				ptr[0] = 0;
				ptr[1] = 0;
				ptr[2] = 0;
			}
		}

		int x=spaceSize,y=spaceSize;
		for(double i : output)
		{
			for(int xx=0; xx<blockSize; xx++)
			{
				for(int yy=0; yy<blockSize; yy++)
				{
					uchar* ptr = mat.ptr(y+yy)+(x+xx)*3;
					ptr[2] = i < 0.5 ? 255-i*512 : 0;
					ptr[1] = i < 0.5 ? i*512 : 255-(i-0.5)*512;
					ptr[0] = i < 0.5 ? 0 : (i-0.5)*512;
					if(xx == 0 || yy == 0 || xx == blockSize-1 || yy == blockSize-1) {ptr[0] = ptr[1] = ptr[2] = 255;}
				}
			}
			x+=blockSize+spaceSize;
			if(x>mat.cols-blockSize-spaceSize)
			{
				y+=blockSize+spaceSize;
				x=spaceSize;
			}
		}

		//Print
		imshow("SigmoidLayer", mat);
		cv::waitKey(10);
	}

	/*
	 *
	 */
	template<ActivationFunction F>
	void SigmoidLayer<F>::printInfo()
	{
		std::cout << "	(" << layerId << ") Sigmoid <-- " << prevLayerId << " : ";
		std::cout << inputSize << " -> " << size << "   w:" << size*(inputSize+1) << "\n";
	}
}
