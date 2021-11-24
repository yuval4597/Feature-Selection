// Yuval Bar 2021

#include "featureselection.h"
#include <iostream>
#include <iomanip>

double FeatureSelection::leaveOneOutCrossValidation(const std::unordered_set<int>& currentFeatures, const int featureToAdd) const
{
	double accuracy = static_cast<double>(rand() / static_cast<double>(RAND_MAX));
	return accuracy;
}

void FeatureSelection::addDataNode(Node node)
{
	dataNodes.push_back(node);

	// If totalFeatures is uninitialized, initialize it with the correct feature size (columns)
	if (totalFeatures.size() < 1)
	{
		totalFeatures.resize(node.features.size(), std::vector<double>());
	}

	// Add this node's features to totalFeatures
	for (auto i = 0; i < totalFeatures.size(); ++i)
	{
		totalFeatures[i].push_back(node.features[i]);
	}
}

void FeatureSelection::printData() const
{
	std::cout << std::scientific << std::setprecision(7);	// Was given numbers with precision of 7

	for (const auto& node : dataNodes)
	{
		std::cout << node.classification << ' ';
		for (const auto& f : node.features)
		{
			std::cout << f << ' ';
		}
		std::cout << '\n';
	}
}

void FeatureSelection::printFeature(const int feature /*= 0*/) const
{
	std::cout << std::scientific << std::setprecision(7);	// Was given numbers with precision of 7

	for (auto i = 0; i < totalFeatures[feature].size(); ++i)
	{
		std::cout << totalFeatures[feature][i] << '\n';
	}
}

void FeatureSelection::featureSearch() const
{
	std::unordered_set<int> currentFeatures;

	for (auto i = 0; i < totalFeatures.size(); ++i)
	{
		std::cout << "On level " << i + 1 << " of the search tree\n";	// i + 1 because index 0
		int featureToAddAtThisLevel = 0;
		double bestSoFarAccuracy = 0.0;

		for (auto j = 0; j < totalFeatures.size(); ++j)
		{
			if (currentFeatures.find(j) != currentFeatures.end())
			{
				// Already added this feature
				continue;
			}

			double accuracy = leaveOneOutCrossValidation(currentFeatures, j);
			std::cout << "--Considering adding feature " << j + 1 << ", accuracy = " << accuracy << '\n';		// j + 1 because index 0
			
			if (accuracy > bestSoFarAccuracy)
			{
				bestSoFarAccuracy = accuracy;
				featureToAddAtThisLevel = j;
			}
		}

		currentFeatures.insert(featureToAddAtThisLevel);
		std::cout << "On level " << i + 1 << " added feature " << featureToAddAtThisLevel + 1 << " to current set\n";	// 1 indexed when printing to the console
	}
}
