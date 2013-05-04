#include <mlpack/core.hpp>
#include "gmm.hpp"

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::util;
using namespace mlpack::kmeans;

int main(int argc, char** argv)
{
    arma::mat data, labels, test;
    
    // Load the data.
    data::Load("../data/mvnrnd_data.csv", data, true);
    
    // Calculate mixture of Gaussians.
    EMFit<> em;    
    GMM<> gmm(2, data.n_rows, em);
    double likelihood = gmm.Estimate(data);
    
    // Vector of covariances; one for each Gaussian.
    Log::cout << gmm.Covariances()[0] << std::endl;
    Log::cout << gmm.Covariances()[1] << std::endl;
    // The vector of means.
    Log::cout << gmm.Means()[0] << std::endl;
    Log::cout << gmm.Means()[1] << std::endl;
    // The a priori weights of each Gaussian.
    Log::cout << gmm.Weights() << std::endl;
    
    return 0;
}