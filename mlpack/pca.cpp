#include <mlpack/core.hpp>
#include "pca.hpp"

using namespace mlpack;
using namespace mlpack::pca;

int main(int argc, char** argv)
{
    arma::mat data, transformedData, coeff;
    arma::vec eigVal;
    
    // Load the data.
    data::Load("../data/ingredients.csv", data, true);
    
    // Perform PCA.
    PCA p;
    p.Apply(data, transformedData, eigVal, coeff);
    
    // Show transform data into eigenvector basis.
    Log::cout << transformedData << std::endl;
    
    return 0;
}