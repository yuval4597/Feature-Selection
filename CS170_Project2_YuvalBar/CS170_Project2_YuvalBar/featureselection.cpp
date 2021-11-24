#include "featureselection.h"
#include <iostream>
#include <iomanip>

void FeatureSelection::addDataNode(Node node)
{
	data.push_back(node);
}

void FeatureSelection::printData() const
{
	std::cout << std::scientific << std::setprecision(7);	// Was given numbers with precision of 7

	for (const auto& node : data)
	{
		std::cout << node.classification << ' ';
		for (const auto& f : node.features)
		{
			std::cout << f << ' ';
		}
		std::cout << '\n';
	}
}

void FeatureSelection::featureSearch(std::vector<Node> data)
{
	
}
