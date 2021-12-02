/*
 * ActivationFunctions.h
 *
 *  Created on: 2 gru 2021
 *      Author: przemo
 */

#ifndef SRC_NEURALNETWORKGPU_LAYER_ACTIVATIONFUNCTIONS_CUH_
#define SRC_NEURALNETWORKGPU_LAYER_ACTIVATIONFUNCTIONS_CUH_

#include"NNLayer.h"

namespace NeuralNetworkGPU
{

	/*
	 *	Default activation function
	 */
	template<ActivationFunction F>
	__device__
	float activationFunctionKernel(float *d_x, float *d_b)
	{
		return __frcp_rd(1 + exp(-(*d_b)*(*d_x)) );
	}

	/*
	 *	Default derivative function
	 */
	template<ActivationFunction F>
	__device__
	float derivativeFunctionKernel(float *d_x, float *d_b)
	{
		float e = __powf(2.71828,(-(*d_b)*(*d_x)));
		float m = 1 + e;
		return __fdiv_rd((*d_b)*e,(m*m));
	}

	/*
	 * Sigmoid activation function
	 */
	template<>
	__device__
	float activationFunctionKernel<ActivationFunction::Sigmoid>(float *d_x, float *d_b)
	{
		return __frcp_rd(1 + exp(-(*d_b)*(*d_x)) );
	}

	/*
	 * Sigmoid derivative function
	 */
	template<>
	__device__
	float derivativeFunctionKernel<ActivationFunction::Sigmoid>(float *d_x, float *d_b)
	{
		float e = __powf(2.71828,(-(*d_b)*(*d_x)));
		float m = 1 + e;
		return __fdiv_rd((*d_b)*e,(m*m));
	}

	/*
	 * Linear activation function
	 */
	template<>
	__device__
	float activationFunctionKernel<ActivationFunction::Linear>(float *d_x, float *d_b)
	{
		return *d_x;
	}

	/*
	 * Linear derivative function
	 */
	template<>
	__device__
	float derivativeFunctionKernel<ActivationFunction::Linear>(float *d_x, float *d_b)
	{
		return 1;
	}

	/*
	 * RELU activation function
	 */
	template<>
	__device__
	float activationFunctionKernel<ActivationFunction::RELU>(float *d_x, float *d_b)
	{
		//fmaxf
		return *d_x > 0 ? *d_x : 0;
	}

	/*
	 * RELU derivative function
	 */
	template<>
	__device__
	float derivativeFunctionKernel<ActivationFunction::RELU>(float *d_x, float *d_b)
	{
		return *d_x > 0 ? 1 : 0;
	}

	/*
	 * LeakRELU activation function
	 */
	template<>
	__device__
	float activationFunctionKernel<ActivationFunction::LeakRELU>(float *d_x, float *d_b)
	{
		//fmaxf
		return *d_x > 0 ? *d_x : *d_x*0.05;
	}

	/*
	 * LeakRELU derivative function
	 */
	template<>
	__device__
	float derivativeFunctionKernel<ActivationFunction::LeakRELU>(float *d_x, float *d_b)
	{
		return *d_x > 0 ? 1 : *d_b;
	}
}


#endif /* SRC_NEURALNETWORKGPU_LAYER_ACTIVATIONFUNCTIONS_CUH_ */
