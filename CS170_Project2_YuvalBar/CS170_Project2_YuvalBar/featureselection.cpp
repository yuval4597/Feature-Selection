// Yuval Bar 2021

#include "featureselection.h"
#include "stats.h"

#include <iostream>
#include <iomanip>
#include <omp.h>

double FeatureSelection::leaveOneOutCrossValidation(std::vector<int> currentFeatures, const int featureToAddOrRemove, SearchType searchType) const
{
	if (searchType == SearchType::ForwardSelection)
	{
		currentFeatures[featureToAddOrRemove] = 1;
	}
	else // searchType == SearchType::BackwardElimination
	{
		currentFeatures[featureToAddOrRemove] = 0;
	}

	int numCorrectlyClassified = 0;

	for (auto i = 0; i < dataNodes.size(); ++i)
	{
		Node objectToClassify = dataNodes[i];

		// Using squared distance to be more performant
		double nearestNeighborSquaredDistance = std::numeric_limits<double>::max();
		int nearestNeighborIndex = 0;

		for (auto j = 0; j < dataNodes.size(); ++j)
		{
			// Don't compare against self
			if (j == i)
			{
				continue;
			}

			// Calculate distance between Node i and Node j
			double squaredDistance = 0.0;
			for (auto k = 0; k < dataNodes[i].features.size(); ++k)
			{
				if (currentFeatures[k] == 0)
				{
					continue;
				}

				double differenceBetweenFeatures = objectToClassify.features[k] - dataNodes[j].features[k];
				squaredDistance += differenceBetweenFeatures * differenceBetweenFeatures;
			}

			if (squaredDistance < nearestNeighborSquaredDistance)
			{
				nearestNeighborSquaredDistance = squaredDistance;
				nearestNeighborIndex = j;
			}
		}

		if (objectToClassify.classification == dataNodes[nearestNeighborIndex].classification)
		{
			++numCorrectlyClassified;
		}
	}

	double accuracy = static_cast<double>(numCorrectlyClassified / static_cast<double>(dataNodes.size()));
	return accuracy;
}

void FeatureSelection::forwardSelection() const
{
	std::vector<int> currentFeatures(totalFeatures.size(), 0);
	std::vector<int> bestFeatures;

	double overallBestAccuracy = 0.0;

	for (auto i = 0; i < totalFeatures.size(); ++i)
	{
		std::cout << "On level " << i + 1 << " of the search tree\n";	// i + 1 because index 0
		int featureToAddAtThisLevel = 0;
		double bestSoFarAccuracy = 0.0;

#if PARALLELIZE
		#pragma omp parallel for
#endif
		for (auto j = 0; j < totalFeatures.size(); ++j)
		{
			if (currentFeatures[j] == 1)
			{
				// Already added this feature
				continue;
			}

			double accuracy = leaveOneOutCrossValidation(currentFeatures, j, SearchType::ForwardSelection);
			std::cout << "--Considering adding feature " << j + 1 << ", accuracy = " << accuracy << " (Thread: " << omp_get_thread_num() << ")\n";		// j + 1 because index 0
			
#if PARALLELIZE
			#pragma omp critical
#endif
			if (accuracy > bestSoFarAccuracy)
			{
				bestSoFarAccuracy = accuracy;
				featureToAddAtThisLevel = j;

				if (bestSoFarAccuracy > overallBestAccuracy)
				{
					overallBestAccuracy = bestSoFarAccuracy;
					bestFeatures = currentFeatures;
					bestFeatures[featureToAddAtThisLevel] = 1;
				}
			}
		}

		currentFeatures[featureToAddAtThisLevel] = 1;
		std::cout << "On level " << i + 1 << " added feature " << featureToAddAtThisLevel + 1 << " to current set\n";	// 1 indexed when printing to the console
	}

	std::cout << "\nBest accuracy = " << overallBestAccuracy << ", with the following features:\n";
	printFeatures(bestFeatures);
	std::cout << std::endl;
}

void FeatureSelection::backwardElimination() const
{
	// In this case starting with all features
	std::vector<int> currentFeatures(totalFeatures.size(), 1);
	std::vector<int> bestFeatures;

	double overallBestAccuracy = 0.0;

	for (int i = totalFeatures.size() - 1; i >= 0; --i)
	{
		std::cout << "On level " << i + 1 << " of the search tree\n";	// i + 1 because index 0
		int featureToRemoveAtThisLevel = 0;
		double bestSoFarAccuracy = 0.0;
		
#if PARALLELIZE
		#pragma omp parallel for
#endif
		for (auto j = 0; j < totalFeatures.size(); ++j)
		{
			if (currentFeatures[j] == 0)
			{
				// Already removed this feature
				continue;
			}

			double accuracy = leaveOneOutCrossValidation(currentFeatures, j, SearchType::BackwardElimination);
			std::cout << "--Considering removing feature " << j + 1 << ", accuracy = " << accuracy << " (Thread: " << omp_get_thread_num() << ")\n";		// j + 1 because index 0

#if PARALLELIZE
			#pragma omp critical
#endif
			if (accuracy > bestSoFarAccuracy)
			{
				bestSoFarAccuracy = accuracy;
				featureToRemoveAtThisLevel = j;

				if (bestSoFarAccuracy > overallBestAccuracy)
				{
					overallBestAccuracy = bestSoFarAccuracy;
					bestFeatures = currentFeatures;
					bestFeatures[featureToRemoveAtThisLevel] = 0;
				}
			}
		}

		currentFeatures[featureToRemoveAtThisLevel] = 0;
		std::cout << "On level " << i + 1 << " removed feature " << featureToRemoveAtThisLevel + 1 << " from current set\n";	// 1 indexed when printing to the console
	}

	std::cout << "\nBest accuracy = " << overallBestAccuracy << ", with the following features:\n";
	printFeatures(bestFeatures);
	std::cout << std::endl;
}

void FeatureSelection::printFeatures(const std::vector<int>& features) const
{
	for (auto i = 0; i < features.size(); ++i)
	{
		if (features[i] == 1)
		{
			std::cout << i + 1 << ' ';
		}
	}
	std::cout << '\n';
}

void FeatureSelection::normalizeData()
{
	std::cout << "Normalizing dataset... ";
	// Get mean and std dev for each feature
	std::vector<double> meanValues(totalFeatures.size(), 0);
	std::vector<double> stdDevValues(totalFeatures.size(), 0);

	for (auto i = 0; i < totalFeatures.size(); ++i)
	{
		Stats s(totalFeatures[i]);
		meanValues[i] = s.getMean();
		stdDevValues[i] = s.getStandardDeviation();
	}

	// Modify data nodes
	for (auto i = 0; i < dataNodes.size(); ++i)
	{
		for (auto j = 0; j < dataNodes[i].features.size(); ++j)
		{
			const double newValue = (dataNodes[i].features[j] - meanValues[j]) / stdDevValues[j];
			dataNodes[i].features[j] = newValue;
			totalFeatures[j][i] = newValue;
		}
	}

	std::cout << "Done.\n";
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

void FeatureSelection::featureSearch(SearchType searchType, bool useNormalizedData /*= false*/)
{
	if (useNormalizedData)
	{
		normalizeData();
	}

	if (searchType == SearchType::ForwardSelection)
	{
		std::cout << "Forward selection...\n";
		forwardSelection();
	}
	else // searchType == SearchType::BackwardElimination
	{
		std::cout << "Backward elimination...\n";
		backwardElimination();
	}
}
