// Yuval Bar 2021

#pragma once
#include <vector>
#include <unordered_set>

enum class SearchType
{
	ForwardSelection,
	BackwardElimination
};

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

	double leaveOneOutCrossValidation(std::unordered_set<int> currentFeatures, const int featureToAddOrRemove, SearchType searchType) const;

	void forwardSelection() const;
	void backwardElimination() const;
public:
	void addDataNode(Node node);

	void printData() const;
	void printFeature(const int feature = 0) const;	// prints all values of this feature from the dataNodes (0 indexed)

	void featureSearch(SearchType searchType) const;
};
