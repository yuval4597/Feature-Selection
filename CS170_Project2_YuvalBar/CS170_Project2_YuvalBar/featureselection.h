// Yuval Bar 2021

#pragma once
#include <vector>
#include <unordered_set>

// Set to 1 for parallel computation, set to 0 for sequential computation
#define PARALLELIZE 0

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

	double leaveOneOutCrossValidation(std::vector<int> currentFeatures, const int featureToAddOrRemove, SearchType searchType) const;

	void forwardSelection() const;
	void backwardElimination() const;
	
	void printFeatures(const std::vector<int>& features) const;

	void normalizeData();
public:
	void addDataNode(Node node);

	void printData() const;
	void printFeature(const int feature = 0) const;	// prints all values of this feature from the dataNodes (0 indexed)

	// Search type is either ForwardSelection or BackwardElimination
	// If useNormalizedData is true, then will normalize data before starting search
	void featureSearch(SearchType searchType, bool useNormalizedData = false);
};
