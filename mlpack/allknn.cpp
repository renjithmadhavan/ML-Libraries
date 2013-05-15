#include <mlpack/core.hpp>
#include "neighbor_search.hpp"
#include "unmap.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;

int main(int argc, char** argv)
{
    // Contains query points.
    arma::mat query(2,3);
    query << 5 << 6 << 2.75 << arma::endr << 1.45 << 2 << 0.75 << arma::endr;

    // Load the data.
    arma::mat data;
    data::Load("../data/fisheriris_data.csv", data, true);
    // Choose the last two columns.
    data = data.rows(2,3);    
    
    std::vector<size_t> oldFromNewRefs;
    std::vector<size_t> oldFromNewQueries;
    arma::mat distancesOut;
    arma::Mat<size_t> neighborsOut;
    
    // Calculate the all 2-nearest-neighbors .
    BinarySpaceTree<bound::HRectBound<2>, QueryStat<NearestNeighborSort> >
    refTree(data, oldFromNewRefs, 20);
    
    BinarySpaceTree<bound::HRectBound<2>, QueryStat<NearestNeighborSort> >*
    queryTree = new BinarySpaceTree<bound::HRectBound<2>,
    QueryStat<NearestNeighborSort> >(query, oldFromNewQueries, 20);
    
    AllkNN* allknn = new AllkNN(&refTree, queryTree, data, query, false);    
    allknn->Search(2, neighborsOut, distancesOut);
    
    arma::Mat<size_t> neighbors;
    arma::mat distances;
    
    // Map the results back to the correct places.
    Unmap(neighborsOut, distancesOut, oldFromNewRefs, oldFromNewQueries,
          neighbors, distances);
    
    // Clean up.
    delete queryTree;
    delete allknn;
    
    // Show the results.
    Log::cout << distances.t() << std::endl;
    Log::cout << neighbors.t() << std::endl;

    return 0;
}