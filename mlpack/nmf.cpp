#include <mlpack/core.hpp>

#include "nmf.hpp"
#include "mult_dist_update_rules.hpp"

using namespace mlpack;
using namespace mlpack::nmf;

int main(int argc, char** argv)
{
    arma::mat V, W, H;

    // Load the data.
    data::Load("../data/fisheriris_data.csv", V, true);
    
    // Compute a nonnegative rank-two approximation
    // of the measurements of the fisher iris data.
    NMF<> nmf;
    nmf.Apply(V, 2, W, H);
    
    // Show the results.
    std::cout << W.t() << std::endl;
    std::cout << H.t() << std::endl;
    std::cout << (W * H).t() << std::endl;

    return 0;
}