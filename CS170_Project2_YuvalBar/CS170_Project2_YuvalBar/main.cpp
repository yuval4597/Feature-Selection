// Yuval Bar 2021

#include "featureselection.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <omp.h>

void addDataFromFile(std::string filename, FeatureSelection& featureSelection)
{
	std::ifstream infile(filename);
	if (!infile.is_open())
	{
		std::cout << "Could not open file " << filename << "!\n";
	}

	if (infile.good())
	{
		std::string inputline;
		while (getline(infile, inputline))
		{
			std::istringstream ss(inputline);
			double val;
			bool isClassSet = false;	// Used to set the first number as class

			Node node;

			while (ss >> val)
			{
				if (!isClassSet)
				{
					node.classification = static_cast<int>(val);
					isClassSet = true;
				}
				else
				{
					node.features.push_back(val);
				}
			}

			featureSelection.addDataNode(node);
		}
	}
	infile.close();
}

int main()
{
	FeatureSelection featureSelection;
	addDataFromFile("Ver_2_CS170_Fall_2021_LARGE_data__22.txt", featureSelection);
	//addDataFromFile("testfile.txt", featureSelection);
	double t0 = omp_get_wtime();
	featureSelection.featureSearch(SearchType::BackwardElimination);
	double t1 = omp_get_wtime();
	std::cout << "Took " << t1 - t0 << " seconds\n";
}