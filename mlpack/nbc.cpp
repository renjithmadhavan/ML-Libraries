#include <mlpack/core.hpp>
#include "naive_bayes_classifier.hpp"

using namespace mlpack;
using namespace mlpack::naive_bayes;

int main(int argc, char** argv)
{
    arma::mat data, labels, test;

    // Load the data.
    data::Load("../data/fisheriris_data.csv", data, true);
    test = data;
    data::Load("../data/fisheriris_label.csv", labels, true);
    data.insert_rows(data.n_rows, labels);
    
    // Naive Bayes classifier.
    NaiveBayesClassifier<> nbc(data, 4);
    
    // Predict labels.
    arma::Col<size_t> results;
    nbc.Classify(test, results);

    Log::cout << results << std::endl;    
    
    return 0;
}
