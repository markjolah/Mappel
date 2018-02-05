/** @file numerical.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 05-2015
 * @brief Numerical matrix operations
 */
#include <cassert>
#include "numerical.h"

namespace mappel {
    
//const double CholeskyDelta = 1e-6; //Minimum we will let a diagonal element be in the modified cholesky algorithm

void copy_Usym_mat(arma::mat &usym)
{
    IdxT size = usym.n_rows;
    if(size != usym.n_cols) ArraySizeError("Expected square matrix");
    for(IdxT j=0;j<size-1;j++) for(IdxT i=j+1;i<size;i++) usym(i,j) = usym(j,i); //i>j
}

void copy_Usym_mat_stack(arma::cube &usym_stack)
{
    IdxT size = usym_stack.n_rows;
    if(size != usym_stack.n_cols) ArraySizeError("Expected stack of square matricies");
    IdxT count= usym_stack.n_slices;
    for(IdxT k=0; k<count;k++) for(IdxT j=0;j<size-1;j++) for(IdxT i=j+1;i<size;i++) 
        usym_stack(i,j,k) = usym_stack(j,i,k); //i>j
}


void copy_Lsym_mat(arma::mat &lsym)
{
    assert(lsym.is_square());
    int size=static_cast<int>(lsym.n_rows);
    for(int j=0;j<size-1;j++) for(int i=j+1;i<size;i++) 
        lsym(j,i) = lsym(i,j); //i>j
}

bool is_negative_definite(const arma::mat &usym)
{
    return is_positive_definite(-usym);
}

bool is_positive_definite(const arma::mat &usym)
{
    arma::mat R=usym;
    return cholesky(R);
}

bool is_symmetric(const arma::mat &A)
{
    if(!A.is_square()) return false;
    for(arma::uword j=1; j<A.n_cols; j++) for(arma::uword i=j+1; i<A.n_rows; i++) if(A(i,j)!=A(j,i)) return false;
    return true;
}

/* Input in in internal cholesky format with D on diagonal and L in lower triangle */
void cholesky_convert_lower_triangular(arma::mat &chol)
{
    int size=static_cast<int>(chol.n_rows);
    for(int j=0; j<size; j++){
        double tmp=sqrt(chol(j,j));
        for(int i=j+1; i<size;i++) chol(i,j)*=tmp;
        chol(j,j) = tmp;
    }
}

/* Input in in internal cholesky format with D on diagonal and L in lower triangle */
void cholesky_convert_full_matrix(arma::mat &chol)
{
    int size=static_cast<int>(chol.n_rows);
    for(int j=1; j<size; j++){
        for(int i=0; i<j;i++) {
            double dotp=chol(j,i)*chol(i,i);
            for(int k=0; k<i; k++) dotp += chol(i,k)*chol(k,k)*chol(j,k); // i>k j>k so read from lower matrix
            chol(i,j)=dotp;   // i<j save to upper tri
        }
    }
    copy_Usym_mat(chol);
}



bool cholesky(arma::mat &A)
{
    //A comes in as upper triangular symmetric, but is converted to a lower triangular
    //Cholesky decomposition format on output.  This way we keep the original matrix arround in
    //the upper triangle while we write the decomposition into the lower triangle.  Cool!
    //We are using the Gill Murray Wright 1981 method although a superior but more complex version
    //bu Schandel and Eskow 1999 exists too that we should eventually implement.
    int size=static_cast<int>(A.n_rows);
    arma::vec v(size-1);
    for(int j=0;j<size;j++) {
        //compute v: 0..j-1
        for(int i=0;i<j;i++) v(i)=A(j,i)*A(i,i); //i<j so use upper triangular access
        double dotp=0;
        for(int i=0;i<j;i++) dotp+=A(j,i)*v(i);
        A(j,j) = A(j,j)-dotp;
//         std::cout<<"J: "<<j<<" dotp:"<<dotp<<" A(j,j)"<<A(j,j)<<"\n";
        if (A(j,j)<=0) return false;
        for(int i=j+1;i<size;i++) {
            double dotp=0;
            for(int k=0;k<j;k++) dotp += A(i,k)*v(k);
            A(i,j)=(A(i,j)-dotp)/A(j,j); //i>j so store to lower triangular
//             std::cout<<"J: "<<j<<" I:"<<i<<" dotp:"<<dotp<<" A(i,j)"<<A(i,j)<<"\n";
        }
    }
    //zero out upper triangular
    for(int j=1;j<size;j++) for(int i=0;i<j;i++)  A(i,j) = 0; //i<j
    return true;
}


bool modified_cholesky(arma::mat &A)
{
    //A comes in as upper triangular symmetric, but is converted to a lower triangular
    //Cholesky decomposition format on output.  This way we keep the original matrix arround in
    //the upper triangle while we write the decomposition into the lower triangle.  Cool!
    //We are using the Gill Murray Wright 1981 method although a superior but more complex version
    //bu Schandel and Eskow 1999 exists too that we should eventually implement.
    assert(A.is_square());
    int size=static_cast<int>(A.n_rows);
    arma::vec v(size);
    double gamma = arma::max(arma::abs(A.diag()));
    double xi = 0;
    for(int j=1;j<size;j++) for(int i=0;i<j;i++) xi = std::max(xi, fabs(A(i,j)));  //Maximum over off-diagonal elements of uppers symmetrix matrix
    double epsilon = std::numeric_limits<double>::epsilon();
    double delta = epsilon*std::max(1.,gamma+xi);
    double beta_sq = std::max(epsilon,std::max(gamma, xi / size));
//     double beta = sqrt(beta_sq);
    bool positive_definite=true;
//     std::cout<<"Modified Cholesky Decomposition.\n";
//     std::cout<<"Size: "<<size<<" Condition:"<<arma::cond(arma::symmatu(A))<<"Lambdas:"<<arma::eig_sym(arma::symmatu(A)).t()<<"\n";
//     std::cout<<"gamma:"<<gamma<<" xi:"<<xi<<" epsilon:"<<epsilon<<" delta:"<<delta<<" beta^2:"<<beta_sq<<" beta:"<<beta<<"\n";
//     std::cout<<"A:\n"<<A;
    for(int j=0;j<size;j++) {
        //compute v: 0..j-1
        for(int i=0;i<j;i++) v(i)=A(j,i)*A(i,i); //i<j so use lower triangular access
//         std::cout<<"v:"<<v.t()<<"\n";
        double dotp=0;
        for(int i=0;i<j;i++) dotp+=A(j,i)*v(i); // lower triangular access to pre-computed
//         std::cout<<"A("<<j<<","<<j<<")="<<A(j,j)<<" dotp:"<<dotp<<std::endl;
        v(j)=A(j,j)-dotp;
        double theta = 0; //Theta will be the max value of the newly computed Cij column j.
        for(int i=j+1;i<size;i++) {
            double dotp=0;
            for(int k=0;k<j;k++) dotp += A(i,k)*v(k);
//             std::cout<<"i:"<<i<<" dotp:"<<dotp<<"\n";
            double val = A(i,j)-dotp; //i>j so store to lower triangular
            theta = std::max(theta, fabs(val));
            A(i,j) = val;
//             std::cout<<"A(i,j)=A("<<i<<","<<j<<")="<<val<<"\n";
        }
        //determine diagonal element
        A(j,j) = std::max(fabs(v(j)), std::max(theta*theta/beta_sq, delta));
//         std::cout<<"v("<<j<<")="<<v(j)<<" theta:"<<theta<<" theta^2/beta^2:"<<theta*theta/beta_sq<<" delta:"<<delta<<"\n";
//         std::cout<<"A("<<j<<","<<j<<")="<<A(j,j)<<"\n";
//         if(j+1<=size-1) std::cout<<"c: "<<A(arma::span(j+1,size-1),j).t()<<"\n";
        if (A(j,j)!=v(j)) {
            positive_definite=false; 
//             std::cout<<"Not positive definite!\n";
        }
        for(int i=j+1;i<size;i++) A(i,j)/=A(j,j); //Renormalize by new diagonal element A(j,j)
//         std::cout<<"A:\n"<<A;
    }
    //zero out upper triangular
    for(int j=1;j<size;j++) for(int i=0;i<j;i++)  A(i,j) = 0; //i<j
    return positive_definite;
}

arma::vec cholesky_solve(const arma::mat &C,const arma::vec &b)
{

    int n=static_cast<int>(C.n_rows);
    arma::vec x(n);
    x=b;
//     std::cout<<"x: "<<x.t()<<"\n";
    //C is lower triangular with D as the diagonal
    //First solve Lx=b (L is unit lower triangular)
    for(int i=1;i<n;i++) {
        double sum=0;
        for(int j=0;j<i;j++) sum+= C(i,j)*x(j);  //i=row,j=col i>j (lower trianglular)
        x(i) -= sum;
    }
//     std::cout<<"x: "<<x.t()<<"\n";
    //Now solve for Dx'=x, store x' in x
    for(int i=0;i<n;i++) x(i)/=C(i,i);
    //Now solve for L^T x'' = x' store in x
//     std::cout<<"x: "<<x.t()<<"\n";
    for(int i=n-2;i>=0;i--) {
        double sum=0;
        for(int j=i+1;j<n;j++) sum+= C(j,i)*x(j);  //j=row,i=col j>i (lower trianglular)
        x(i) -= sum;
    }
//     std::cout<<"x: "<<x.t()<<"\n";
    return x;
}

} /* namespace mappel */
    
    
