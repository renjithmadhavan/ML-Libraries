#include <mlpack/core.hpp>
#include "kmeans.hpp"

using namespace mlpack;
using namespace mlpack::kmeans;

int main(int argc, char** argv)
{
    arma::mat data;
    
    // Load the data.
    data::Load("../data/two_cluster.csv", data, true);
    
    // Perform kmean with 2 clusters.
    arma::Col<size_t> idx;
    KMeans<> k;
    arma::mat ctrs;
    k.Cluster(data, 2, idx, ctrs);
    
    // Show cluster association.
    Log::cout << idx << std::endl;
    
    // Show cluster centers.
    Log::cout << ctrs << std::endl;
}