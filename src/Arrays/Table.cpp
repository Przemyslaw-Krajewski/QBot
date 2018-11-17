/*
 * Table.cpp
 *
 *  Created on: 18 lip 2018
 *      Author: przemo
 */

#include "Table.h"

/*
 *
 */
Table::Table(int t_nActions, std::vector<int> t_dimensionsSize)
{
	table.clear();

	long long size = 1;
	for(int i=0; i<t_dimensionsSize.size(); i++) size *= t_dimensionsSize[i];

	std::vector<double> row(size,1.0);

	for(int i=0; i<t_nActions; i++)
	{
		table.push_back(row);
	}
	dimensionsSize = t_dimensionsSize;
}

/*
 *
 */
Table::~Table()
{

}

/*
 *
 */
double Table::getValue(std::vector<int> t_state, int t_action)
{
	long long index = 0;
	long long multiplier = 1;

	for(int i=0; i<t_state.size(); i++)
	{
		index += t_state[i] * multiplier;
		multiplier *= dimensionsSize[i];
	}
	return table[t_action][index];
}

/*
 *
 */
std::vector<double> Table::getValues(std::vector<int> t_state)
{
	long long index = 0;
	long long multiplier = 1;

	for(int i=0; i<t_state.size(); i++)
	{
		index += t_state[i] * multiplier;
		multiplier *= dimensionsSize[i];
	}

	std::vector<double> result;
	for(int i=0; i<4; i++) result.push_back(table[i][index]);

	return result;
}

/*
 *
 */
void Table::setValue(std::vector<int> t_state, int t_action, double t_value)
{
	long long index = 0;
	long long multiplier = 1;

	for(int i=0; i<t_state.size(); i++)
	{
		index += t_state[i] * multiplier;
		multiplier *= dimensionsSize[i];
	}

	table[t_action][index] = t_value;
}
