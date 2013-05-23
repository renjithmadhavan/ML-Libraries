#include <mlpack/core.hpp>
#include "lars.hpp"

using namespace mlpack;
using namespace mlpack::regression;

int main(int argc, char** argv)
{
    arma::mat X, Y;
    
    // Load the data.
    data::Load("../data/artificial_data_X.csv", X, true, false);
    data::Load("../data/artificial_data_Y.csv", Y, true, false);
    
    // Perform LARS.
    LARS lars(0, 0, false);
    arma::vec beta;
    lars.Regress(X, Y.unsafe_col(0), beta, false);
    
    // Show the results.
    std::cout << beta << std::endl;
    
    return 0;
}