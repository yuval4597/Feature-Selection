#pragma once
#include <vector>

struct Node
{
	int classification;	// 0 or 1
	std::vector<double> features;
};

class FeatureSelection
{
public:
	static void featureSearch(std::vector<Node> data);
};
