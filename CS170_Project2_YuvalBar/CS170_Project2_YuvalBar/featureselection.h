#pragma once
#include <vector>

struct Node
{
	int classification;	// 0 or 1
	std::vector<double> features;
};

class FeatureSelection
{
private:
	std::vector<Node> dataNodes;

public:
	void addDataNode(Node node);
	void printData() const;
	void featureSearch() const;
};
