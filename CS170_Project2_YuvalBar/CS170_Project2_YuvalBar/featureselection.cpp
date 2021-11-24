#include "featureselection.h"
#include <iostream>
#include <iomanip>

void FeatureSelection::addDataNode(Node node)
{
	dataNodes.push_back(node);
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

void FeatureSelection::featureSearch() const
{
	for (auto i = 0; i < dataNodes.size(); ++i)
	{
		std::cout << "On level " << i + 1 << " of the search tree\n";	// i + 1 because index 0

		for (auto j = 0; j < dataNodes.size(); ++j)
		{
			std::cout << "--Considering adding feature " << j + 1 << '\n';		// j + 1 because index 0		
		}
	}
}
