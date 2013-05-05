#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include "range_search.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::tree;

typedef RangeSearch<metric::SquaredEuclideanDistance,
BinarySpaceTree<bound::HRectBound<2>, EmptyStatistic> > RSType;

int main(int argc, char** argv)
{
    arma::mat data, queryData;
    vector<vector<size_t> > neighbors;
    vector<vector<double> > distances;
    
    // Load the data.
    data::Load("../data/rnd_points_large.csv", data, true);
    data::Load("../data/rnd_points_small.csv", queryData, true);

    // Mappings for when we build the tree.
    vector<size_t> oldFromNewRefs;
    std::vector<size_t> oldFromNewQueries;

    // Find all neighbors within specified distance using KDTreeSearcher object
    BinarySpaceTree<bound::HRectBound<2>, tree::EmptyStatistic>
    refTree(data, oldFromNewRefs, 20);
    
    BinarySpaceTree<bound::HRectBound<2>, tree::EmptyStatistic>*
    queryTree = new BinarySpaceTree<bound::HRectBound<2>,
    tree::EmptyStatistic >(queryData, oldFromNewQueries, 20);

    RSType* rangeSearch = new RSType(&refTree, queryTree, data, queryData, false);

    math::Range r = math::Range(0, 0.1);
    rangeSearch->Search(r, neighbors, distances);
    
    // This is only capable for this data set.
    // Show the results.
    cout << neighbors[0][0] << endl;
    cout << neighbors[1][0] << endl;
    
    cout << distances[0][0] << endl;
    cout << distances[1][0] << endl;
    
    return 0;
}