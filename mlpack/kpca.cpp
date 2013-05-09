#include <mlpack/core.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include "kernel_pca.hpp"

using namespace mlpack;
using namespace mlpack::kpca;
using namespace mlpack::kernel;

int main(int argc, char** argv)
{
    arma::mat data, transformedData, eigvec;
    arma::vec eigVal;
    
    // Load the data.
    data::Load("../data/circle_data.txt", data, true);
    
    // Using the Gaussian Kernel to construct the Kernel Matrix.
    GaussianKernel kernel;    
    arma::mat kernelMat = GetKernelMatrix(kernel, trans(data));
    
    // For PCA the data has to be centered, even if the data is centered.
    // It is nit guarantee the data when mapped is also centered. Since
    // we actually never work in the feature space we cannot center the data.
    // Since centered data is required to perform an effective principal
    // component analysis we perform a pseudo center method using the Kernel Matrix.
    arma::mat oneMat = arma::ones<arma::mat>(kernelMat.n_rows, kernelMat.n_cols);
    arma::mat kernelMatCenter = kernelMat - oneMat*kernelMat - kernelMat*oneMat + oneMat*kernelMat*oneMat;
    
    // Compute eigenvectors and the corresponding eigenvalues.
    arma::eig_sym(eigVal, eigvec, kernelMatCenter);
    
    // The eigenvectors and the corresponding eigenvalues are
    // already sorted but in the wrong order.
    // Since descend is required, we reverse the eigenvectors
    // and the corresponding eigenvalues.
    // To avoid temporary matrices we use swap.
    int n_eigVal = eigVal.n_elem;
    for(int i = 0; i < floor(n_eigVal / 2.0); i++)
        eigVal.swap_rows(i, (n_eigVal - 1) - i);
    
    eigvec = arma::fliplr(eigvec);
    
    // Dimension of output data.
    size_t dim = 2;
    
    // Projecting the data in lower dimensions.
    transformedData = eigvec.submat(0, 0, eigvec.n_rows-1, dim-1).t() * kernelMatCenter.t();
    
    std::cout << transformedData.t() << std::endl;
    
    return 0;
}