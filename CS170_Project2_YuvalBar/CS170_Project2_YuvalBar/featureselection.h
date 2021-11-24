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
	std::vector<std::vector<double>> totalFeatures;

public:
	void addDataNode(Node node);

	void printData() const;
	void printFeature(const int feature = 0) const;	// prints all values of this feature from the dataNodes (0 indexed)

	void featureSearch() const;
};
