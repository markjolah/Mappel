/*!
 * \file cGaussMLE.h
 * \author Keith Lidke
 * \date January 10, 2010
 * \brief The MLE optimization routines 
 */

#ifndef _CGAUSSMLE_H
#define _CGAUSSMLE_H
#include <armadillo>

namespace mappel {
namespace cgauss {
using FVecT = arma::Col<float>;
using VecT = arma::Col<double>;
using MatT = arma::Mat<double>;

template<class FType>
FVecT convertToCGaussCoords(const arma::Col<FType> &theta)
{
    FVecT ftheta = arma::conv_to<FVecT>::from(theta);
    if(!ftheta.is_empty()){
        float temp = ftheta(0)-.5;
        ftheta(0) = ftheta(1)-.5;
        ftheta(1) = temp;
    }
    return ftheta;
}

template<class FType>
VecT convertFromCGaussCoords(const arma::Col<FType> &theta)
{
    arma::Col<double> dtheta = arma::conv_to<arma::Col<double>>::from(theta);
    double temp = dtheta(0)+.5;
    dtheta(0) = dtheta(1)+.5;
    dtheta(1) = temp;
    return dtheta;
}

template<class FType>
MatT convertFromCGaussCoords(const arma::Mat<FType> &theta)
{
    int N=static_cast<int>(theta.n_cols);
    MatT dtheta = arma::conv_to<MatT>::from(theta);
    for(int n=0; n<N; n++){
        double temp = dtheta(0,n)+.5;
        dtheta(0,n) = dtheta(1,n)+.5;
        dtheta(1,n) = temp;
    }
    return dtheta;
}

template<class FType>
FVecT convertToCGaussCoords_sigma(const arma::Col<FType> &theta, double min_sigma)
{
    auto ftheta = convertToCGaussCoords(theta);
    ftheta(4) *= min_sigma;
    return ftheta;
}

template<class FType>
VecT convertFromCGaussCoords_sigma(const arma::Col<FType> &theta, double min_sigma)
{
    auto dtheta = convertFromCGaussCoords(theta);
    dtheta(4) /= min_sigma;
    return dtheta;
}

template<class FType>
MatT convertFromCGaussCoords_sigma(const arma::Mat<FType> &theta, double min_sigma)
{
    auto dtheta = convertFromCGaussCoords(theta);
    dtheta.row(4) /= min_sigma;
    return dtheta;
}

/* 4-parameter fixed-sigma model where 5th parameter is sigma measured in pixels */
void MLEInit(const float data[], float PSFSigma, int size, float theta_est[]);

void MLEFit(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
            float theta_est[]);

void MLEFit_debug(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
                  float theta_est[], MatT &sequence);

/** Original cGauss MLEFit code in full with adaptation to theta_init
 */
void MLEFit_full(const float data[], const float PSFSigma, const int size,
                 const int iterations, const FVecT &theta_init, float theta_est[], float CRLB[],
                 float *LogLikelihood);


/* 5-parameter sigma model where 5th parameter is sigma measured in pixels */
void MLEInit_sigma(const float data[], float PSFSigma, int size, float theta_est[]);

void MLEFit_sigma(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
                  float theta_est[]);

void MLEFit_sigma_debug(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
                        float theta_est[], MatT &sequence);

/** Original cGauss MLEFit_sigma code in full with adaptation to theta_init
 */
void MLEFit_sigma_full(const float data[], const float PSFSigma, const int size,
                 const int iterations, const FVecT &theta_init, float theta_est[], float CRLB[],
                 float *LogLikelihood);



// void MLEFit_sigmaxy(const int subregion, const float *d_data, 
//                     const float PSFSigma, const int sz, const int iterations, 
//                     float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,
//                     const int Nfits);



// void MLEFit_sigmaxy_debug(const int subregion, const float *d_data, 
//                     const float PSFSigma, const int sz, const int iterations, 
//                     float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,
//                     const int Nfits, const MatT &sequence, const StatsT &stats);


// 
// void MLEFit_z(const float *d_data, const float PSFSigma_x, const float Ax, 
//               const float Ay, const float Bx, const float By, const float gamma,
//               const float d, const float PSFSigma_y, const int sz, 
//               const int iterations, float *d_Parameters, float *d_CRLBs, 
//               float *d_LogLikelihood);
// 

} /* namespace cgauss */
} /* namespace mappel */

#endif /* _CGAUSSMLE_H */
