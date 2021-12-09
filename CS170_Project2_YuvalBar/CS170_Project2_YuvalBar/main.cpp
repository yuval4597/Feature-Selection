// Yuval Bar 2021

#include "featureselection.h"

#include <iostream>
#include <omp.h>

int main()
{
	std::cout << "What file would you like to load the dataset from (include extension)? ";

	std::string inputFilename;
	std::cin >> inputFilename;

	FeatureSelection featureSelection(inputFilename);

	bool normalizeData = true;

	std::cout << "Would you like to normalize the data (1 - normalize/ 0 - do NOT normalize)? ";
	std::cin >> normalizeData;

	int searchInput = 0;

	std::cout << "Forward selection (0) or backward elimination (1)? ";
	std::cin >> searchInput;

	SearchType searchType;
	switch (searchInput)
	{
	case 0:
		searchType = SearchType::ForwardSelection;
		break;
	case 1:
		searchType = SearchType::BackwardElimination;
		break;
	default:
		searchType = SearchType::ForwardSelection;;
	}

	const bool createOutputFile = true;

	featureSelection.featureSearch(searchType, normalizeData, createOutputFile);
}