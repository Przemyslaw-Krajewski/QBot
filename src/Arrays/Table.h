/*
 * Table.h
 *
 *  Created on: 18 lip 2018
 *      Author: przemo
 */

#ifndef SRC_ARRAYS_TABLE_H_
#define SRC_ARRAYS_TABLE_H_

#include <iostream>
#include <vector>

#include "Array.h"

class Table : public Array
{
public:
	Table(int t_nActions, std::vector<int> t_dimensionsSize);
	virtual ~Table();

	virtual double getValue(std::vector<int> t_state, int t_action);
	virtual std::vector<double> getValues(std::vector<int> t_state);
	virtual void setValue(std::vector<int> t_state, int t_action, double t_value);

	std::vector<std::vector<double>> table;
	std::vector<int> dimensionsSize;
};

#endif /* SRC_ARRAYS_TABLE_H_ */
