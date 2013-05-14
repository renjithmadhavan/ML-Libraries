#include <mlpack/core.hpp>
#include "radical.hpp"

using namespace mlpack;
using namespace mlpack::radical;
using namespace mlpack::math;

int main(int argc, char** argv)
{
    arma::mat data, Y, W;
    
    // Load the data.
    data::Load("../data/radical.csv", data);
    
    // Run RADICAL.
    RandomSeed((size_t) std::time(NULL));
    Radical rad(0.175, 30, 150, data.n_rows-1);
    rad.DoRadical(data, Y, W);
    
    Log::cout << Y.t() << std::endl;
    Log::cout << W.t() << std::endl;
    
    return 0;
}