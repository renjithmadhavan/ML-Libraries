#include <mlpack/core.hpp>
#include "dtb.hpp"

using namespace mlpack;
using namespace mlpack::emst;

int main(int argc, char** argv)
{
    arma::mat data, output;
    // Load the data.
    data::Load("../data/emst.csv", data, true);
    
    // Compute the Euclidean minimum spanning tree
    // using the dual-tree Boruvka algorithm.
    DualTreeBoruvka<> dtb(data);
    dtb.ComputeMST(output);
    
    // Show the results.
    std::cout << output.t() << std::endl;
    
    return 0;
}