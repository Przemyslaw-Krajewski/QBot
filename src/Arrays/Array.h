/*
 * Array.h
 *
 *  Created on: 18 lip 2018
 *      Author: przemo
 */

#ifndef SRC_ARRAYS_ARRAY_H_
#define SRC_ARRAYS_ARRAY_H_

#include <vector>
#include <assert.h>

class Array {
public:
	Array();
	virtual ~Array();

	virtual double getValue(std::vector<int> state, int action) = 0;
	virtual std::vector<double> getValues(std::vector<int> state) = 0;
	virtual void setValue(std::vector<int> state, int action, double value) = 0;

	virtual void printInfo() {}
};

#endif /* SRC_ARRAYS_ARRAY_H_ */
