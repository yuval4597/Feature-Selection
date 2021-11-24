#pragma once
#include <vector>

struct Node
{
	int classification;	// 0 or 1
	std::vector<long double> features;
};

class FeatureSelection
{
private:
	std::vector<Node> data;

public:
	void addDataNode(Node node);
	void printData() const;
	static void featureSearch(std::vector<Node> data);
};
