/** @file numerical.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 05-22-2015
 * @brief Numerical matrix operations
 */
#ifndef MAPPEL_NUMERICAL_H
#define MAPPEL_NUMERICAL_H

#include <cmath>
#include <climits>
#include <armadillo>
#include "Mappel/util.h"

namespace mappel {
    
/** 
 * Convert symmetric matrix stored as upper triangular to full Matrix
 * Assuming usym is the main diagonal and upper triangle of a symmertic matrix, fill in the lower
 * triangle by copying the upper triangle.  This operation modifies the matrix.
 * 
 */
void copy_Usym_mat(arma::mat &usym);

void copy_Usym_mat_stack(arma::cube &usym_stack);

/** 
 * Convert symmetric matrix stored as lower triangular to full Matrix
 * Assuming lsym is the main diagonal and lower triangle of a symmertic matrix, fill in the upper
 * triangle by copying the lowerr triangle.  This operation modifies the matrix.
 * 
 */
void copy_Lsym_mat(arma::mat &lsym);


/** Convert matrix in internal cholesky format into a lower triangular matrix L where M = L*L' */
void cholesky_convert_lower_triangular(arma::mat &chol);

/** Convert matrix in internal cholesky format into a full matrix M = L*L' */
void cholesky_convert_full_matrix(arma::mat &chol);

/** Modify m inplace using modfied choslesky decomposition to ensure m is negative definite */
void cholesky_make_negative_definite(arma::mat &m);

/** Modify m inplace using modfied choslesky decomposition to ensure m is positive definite */
void cholesky_make_positive_definite(arma::mat &m);


/**
 * Determine if C is positive definite
 * @param usym A symmetric matrix in upper triangular format.
 * @return True if C is positive definite
 */
bool is_positive_definite(const arma::mat &usym);

/**
 * Determine if C is negative definite (i.e., -C is positive definite)
 * @param usym A symmetric matrix in upper triangular format.
 * @return True if C is negative definite
 */
bool is_negative_definite(const arma::mat &usym);


/**
 * Check that full 2D matrix A is symmetric and can thus be treated as either 
 * upper or lower triangular symmetric representation.  This will obviously not
 * work with matricies that are already implicitly stored as symmetric triangular format
 * since those matricies won't have the other triangle of elements filled in correctly.
 * 
 * 
 */
bool is_symmetric(const arma::mat &A);


/**
 * @param usym An upper triangular symmetric matrix stored in a full matrix format.
 *             This matrix will be overwritten with the upper triangle and diagonal elements of the
 *             modified cholesky decomposition.
 * @return true if usym was positive semi-definite.  If false then Usym is left in arbitrary corrupted state.
 */
bool cholesky(arma::mat &usym);


/**
 * @param usym An upper triangular symmetric matrix stored in a full matrix format.
 *             This matrix will be overwritten with the upper triangle and diagonal elements of the
 *             modified cholesky decomposition.
 * @return true if usym was positive semi-definite (no cholesky modification required).  If false
 *         we made a modification
 */
bool modified_cholesky(arma::mat &usym);

/**
 * Given a matrix in modified cholesky format and a vector solve the linear system
 * C x = b.
 * @param C A matrix in lower modified cholesky format
 * @param b A vector representing the right hand side of the linear system.
 * @return x - the solution to the linear system
 */
arma::vec cholesky_solve(const arma::mat &C,const arma::vec &b);

} /* namespace mappel */

#endif /* MAPPEL_NUMERICAL_H*/
