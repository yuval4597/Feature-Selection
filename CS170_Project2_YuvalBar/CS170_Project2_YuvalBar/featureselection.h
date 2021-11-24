#pragma once
#include <vector>

struct Node
{
	int classification;	// 0 or 1
	std::vector<double> features;
};

// Hash function for the unordered_set of Nodes used in certain FeatureSelection functions
struct HashFn
{
	size_t operator()(const Node& n) const
	{
		size_t res = std::hash<double>{}(n.features[0]);
		return res;
	}
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
