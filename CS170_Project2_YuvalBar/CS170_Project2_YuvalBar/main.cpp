// Yuval Bar 2021

#include "featureselection.h"

#include <iostream>
#include <omp.h>

int main()
{
	FeatureSelection featureSelection("Ver_2_CS170_Fall_2021_Small_data__75.txt");

	const bool normalizeData = true;
	const bool createOutputFile = true;

	double t0 = omp_get_wtime();
	featureSelection.featureSearch(SearchType::ForwardSelection, normalizeData, createOutputFile);
	double t1 = omp_get_wtime();

	std::cout << "Took " << t1 - t0 << " seconds\n";
}