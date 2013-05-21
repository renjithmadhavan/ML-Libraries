#include <mlpack/core.hpp>
#include "nca.hpp"

using namespace mlpack;
using namespace mlpack::nca;
using namespace mlpack::metric;
using namespace mlpack::optimization;

int main(int argc, char** argv)
{
    arma::mat data;
    // Load the Data.
    data::Load("../data/fisheriris_data.csv", data, true);
    // Load the labels.
    arma::umat labels(data.n_cols, 1);
    data::Load("../data/fisheriris_label.csv", labels, true);
    arma::uvec label = labels.unsafe_col(0).t();

    // Create the NCA object and run the optimization.
    NCA<LMetric<2> > nca(data, label);
    nca.Optimizer().StepSize() = 0.01;
    nca.Optimizer().MaxIterations() = 200;
    nca.Optimizer().Tolerance() = 1/10000000;
    nca.Optimizer().Shuffle() = true;
    
    arma::mat outputMatrix;
    nca.LearnDistance(outputMatrix);
    
    // Show the results.
    std::cout << outputMatrix << std::endl;     
    
    return 0;
}