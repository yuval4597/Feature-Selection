// Yuval Bar 2021

#pragma once
#include <vector>
#include <fstream>

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
	bool isDataValid = false;	// If data was successfully initialized from a file, this will be true

	std::vector<Node> dataNodes;
	std::vector<std::vector<double>> totalFeatures;

	std::ofstream outfile;	// Used to show information regarding num of features and accuracies (.txt file)
	std::string infileName;	// Name of file data came from (set in initDataFromFile)

	double leaveOneOutCrossValidation(std::vector<int> currentFeatures, const int featureToAddOrRemove, SearchType searchType) const;

	void forwardSelection();
	void backwardElimination();
	
	void printFeatures(const std::vector<int>& features) const;					// Prints features from vector of 1s (feature included) and 0s (feature not included)
	std::string getFeaturesAsString(const std::vector<int>& features) const;	// Returns string of features included in this vector (since vector stores 1s and 0s)

	void createOutputFile();

	void normalizeData();
public:
	void addDataNode(Node node);
	void initDataFromFile(std::string filename);

	void printData() const;
	void printFeature(const int feature = 0) const;	// Prints all values of this feature from the dataNodes (0 indexed)

	void printDatasetInfo();	// Prints info about the dataset like number of instances and number of features

	// Search type is either ForwardSelection or BackwardElimination
	// If useNormalizedData is true, then will normalize data before starting search
	// If createOutputFile is true then information regarding the search will be printed to an output .txt file
	void featureSearch(SearchType searchType, bool useNormalizedData = false, bool createOutputFile = false);

	FeatureSelection();
	FeatureSelection(std::string filename);
	~FeatureSelection();
};
