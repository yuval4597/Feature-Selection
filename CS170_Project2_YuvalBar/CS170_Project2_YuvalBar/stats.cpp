// Yuval Bar 2021

#include "stats.h"

#include <iostream>
#include <fstream>
#include <algorithm>

Stats::Stats(std::vector<double> inData)
{
	data = inData;
}

void Stats::printData(int numPerRow /* = 10 */)
{
	int untilNextRow = numPerRow;

	for (const auto& num : data)
	{
		std::cout << num << '\t';
		--untilNextRow;
		if (untilNextRow <= 0) {
			untilNextRow = numPerRow;
			std::cout << '\n';
		}
	}

	if (untilNextRow != numPerRow) {
		std::cout << '\n';
	}
}

void Stats::sortData(bool ascendingOrder /* = true */)
{
	if (ascendingOrder)
	{
		sort(data.begin(), data.end());
		return;
	}

	sort(data.begin(), data.end(), [](const auto& a, const auto& b) { return b < a; });
}

double Stats::getMean()
{
	double sum = 0.0;
	for (const auto& num : data)
	{
		sum += num;
	}

	return sum / data.size();
}

double Stats::getMedian()
{
	if (data.size() == 1)
	{
		return data[0];
	}

	std::vector<double> tempData = data;
	sort(tempData.begin(), tempData.end());

	if (tempData.size() % 2 == 0)
	{
		int posIndex = tempData.size() / 2 - 1;
		return (tempData[posIndex] + tempData[posIndex + 1]) / 2;
	}
	else
	{
		int posIndex = tempData.size() / 2;
		return tempData[posIndex];
	}
}

double Stats::getStandardDeviation()
{
	double mean = getMean();
	double sum = 0;

	for (const auto& num : data)
	{
		sum += pow(num - mean, 2);
	}

	return sqrt(sum / (data.size() - 1));
}
