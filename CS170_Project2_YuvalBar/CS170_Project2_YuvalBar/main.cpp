#include "featureselection.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

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
			long double val;
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
	addDataFromFile("CS170_Fall_2021_SMALL_data__86.txt", featureSelection);
	featureSelection.printData();
}