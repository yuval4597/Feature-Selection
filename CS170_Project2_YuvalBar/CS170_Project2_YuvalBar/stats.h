// Yuval Bar 2021

#pragma once

#include <vector>
#include <string>

class Stats
{
private:
	std::vector<double> data;

public:
	Stats(std::vector<double> inData);

	void printData(int numPerRow = 10);

	// Ascending order if true, descending if false
	void sortData(bool ascendingOrder = true);

	double getMean();
	double getMedian();
	double getStandardDeviation();
};