
/** @file stencil.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 03-22-2014
 * @brief The stencils for pixel based computations
 */
#ifndef _STENCIL_H
#define _STENCIL_H

#include <cmath>
#include <climits>
#include "util.h"
#include "rng.h"
// #include <boost/math/pdf.hpp>

namespace mappel {
static const double sqrt2= sqrt(2);
static const double sqrt2pi= sqrt(2*arma::datum::pi);

double gauss_norm(double sigma);

void fill_d_stencil(int size, double stencil[], double theta_x);
void fill_G_stencil(int size, double stencil[], const double dx[], double theta_sigma);
void fill_X_stencil(int size, double stencil[], const double dx[], double theta_sigma);
void fill_DX_stencil(int size, double stencil[], const double Gx[], double theta_sigma);
void fill_DXS_stencil(int size, double stencil[], const double dx[], const double Gx[], double theta_sigma);
void fill_DXS2_stencil(int size, double stencil[], const double dx[], const double Gx[], const double DXS[], double theta_sigma);
void fill_DXSX_stencil(int size, double stencil[], const double dx[], const double Gx[], const double DX[], double theta_sigma);

VecT make_d_stencil(int size, double theta_x);
VecT make_G_stencil(int size, const VecT &dx, double theta_sigma);
VecT make_X_stencil(int size, const VecT &dx, double theta_sigma);
VecT make_DX_stencil(int size, const VecT &Gx, double theta_sigma);
VecT make_DXS_stencil(int size, const VecT &dx, const VecT &Gx, double theta_sigma);
VecT make_DXS2_stencil(int size, const VecT &dx, const VecT &Gx, const VecT &DXS, double theta_sigma);
VecT make_DXSX_stencil(int size, const VecT &dx, const VecT &Gx, const VecT &DX, double theta_sigma);

void fill_gaussian_stencil(int size, double stencil[], double sigma);
VecT make_gaussian_stencil(int size, double sigma);

void estimate_gaussian_2Dmax(const MatT &data, const VecT &Xstencil,const VecT &Ystencil, int max_pos[], double &min_val);
void refine_gaussian_2Dmax(const MatT &data, const VecT &Xstencil,const VecT &Ystencil, int max_pos[]);
double gaussian_convolution(int x, int y, const MatT &data, const VecT &Xstencil,const VecT &Ystencil);

void estimate_gaussian_3Dmax(const CubeT &data, const VecFieldT &stencils, int max_pos[], double &min_val);
void refine_gaussian_3Dmax(const CubeT &data, const VecFieldT &stencils, int max_pos[]);
double gaussian_3D_convolution(int x, int y, int z, const CubeT &data, const VecFieldT &stencils);

double poisson_log_likelihood(double model_val, double data_val);
double relative_poisson_log_likelihood(double model_val, double data_val);

double check_lower_bound_hyperparameter(const char* name, double value, double lower_bound);
double check_positive_hyperparameter(const char* name, double value, double hyperprior_epsilon=1E-6);
double check_unit_hyperparameter(const char* name, double value, double hyperprior_epsilon=1E-6);

double log_prior_beta_const(double beta);
double log_prior_beta2_const(double beta0, double beta1);
double log_prior_gamma_const(double kappa, double mean);
double log_prior_pareto_const(double alpha, double min);
double log_prior_normal_const(double sigma);


double rllh_beta_prior(double beta, double v, double max=1., double min=0.);
double rllh_beta2_prior(double beta0, double beta1, double v, double max=1., double min=0.);
double rllh_gamma_prior(double kappa, double mean, double v);
double rllh_pareto_prior(double alpha, double v);
double rllh_normal_prior(double mu, double sigma);

double beta_prior_grad(double beta, double v, double max=1., double min=0.);
double beta2_prior_grad(double beta0, double beta1, double v, double max=1., double min=0.);
double gamma_prior_grad(double kappa, double mean, double v);
double pareto_prior_grad(double alpha,  double v);
double normal_prior_grad(double mu, double sigma);

double beta_prior_grad2(double beta, double v, double max=1., double min=0.);
double beta2_prior_grad2(double beta0, double beta1, double v, double max=1., double min=0.);
double gamma_prior_grad2(double kappa, double v);
double pareto_prior_grad2(double alpha,  double v);
double normal_prior_grad(double sigma);
/* Functions to make candidate distributions */
// template<class rng_t>
// double sample_beta_cand_dist(rng_t &rng, double &val, double min_val, double max_val, double eta=10.);
// 
// template<class rng_t>
// double sample_gamma_cand_dist(rng_t &rng, double &val, double min_val, double eta=10.);

double normal_quantile_twosided(double confidence);
double normal_quantile_onesided(double confidence);


/* inline function definitions */
inline double gauss_norm(double sigma)
{ 
    return 1./(sigma*sqrt2);
}

inline VecT make_d_stencil(int size, double theta_x)
{
    VecT stencil(size+1);
    fill_d_stencil(size, stencil.memptr(), theta_x);
    return stencil;
}

inline VecT make_G_stencil(int size, const VecT &dx, double theta_sigma)
{
    VecT stencil(size+1);
    fill_G_stencil(size, stencil.memptr(), dx.memptr(), theta_sigma);
    return stencil;
}

inline VecT make_X_stencil(int size, const VecT &dx, double theta_sigma)
{
    VecT stencil(size);
    fill_X_stencil(size, stencil.memptr(), dx.memptr(), theta_sigma);
    return stencil;
}

inline VecT make_DX_stencil(int size, const VecT &Gx, double theta_sigma)
{
    VecT stencil(size);
    fill_DX_stencil(size, stencil.memptr(), Gx.memptr(), theta_sigma);
    return stencil;
}

inline VecT make_DXS_stencil(int size, const VecT &dx,
                                const VecT &Gx, double theta_sigma)
{
    VecT stencil(size);
    fill_DXS_stencil(size, stencil.memptr(), dx.memptr(), Gx.memptr(), theta_sigma);
    return stencil;
}

inline VecT make_DXS2_stencil(int size, const VecT &dx,
                const VecT &Gx, const VecT &DXS, double theta_sigma)
{
    VecT stencil(size);
    fill_DXS2_stencil(size, stencil.memptr(), dx.memptr(), Gx.memptr(), DXS.memptr(), theta_sigma);
    return stencil;
}

inline VecT make_DXSX_stencil(int size, const VecT &dx,
                const VecT &Gx, const VecT &DX, double theta_sigma)
{
    VecT stencil(size);
    fill_DXSX_stencil(size, stencil.memptr(), dx.memptr(), Gx.memptr(), DX.memptr(), theta_sigma);
    return stencil;
}


inline
void fill_d_stencil(int size, double stencil[], double theta_x)
{
    for(int i=0;i<=size;i++) stencil[i]=i-theta_x;
}

inline
void fill_G_stencil(int size, double stencil[], const double dx[], double theta_sigma)
{
    double norm = -.5/square(theta_sigma);
    for(int i=0;i<=size;i++)  stencil[i] = exp(norm*square(dx[i]));
}

inline
void fill_X_stencil(int size, double stencil[], const double dx[], double theta_sigma)
{
    double norm = 1./sqrt2/theta_sigma;
    double derf = erf(norm*(dx[0]));
    for(int i=0;i<size;i++) {
        double last_derf = derf;
        derf = erf(norm*(dx[i+1]));
        stencil[i] = 0.5*(derf-last_derf);
    }
}

inline
void fill_DX_stencil(int size, double stencil[],const double Gx[], double theta_sigma)
{
    double norm = 1./sqrt2pi/theta_sigma; /* 1/(sqrt(2*pi)*sigma) */
    for(int i=0;i<size;i++) stencil[i] = norm*(Gx[i]-Gx[i+1]);
}

inline
void fill_DXS_stencil(int size, double stencil[], const double dx[], const double Gx[], double theta_sigma)
{
    double norm = 1./sqrt2pi/square(theta_sigma); /* 1/(sqrt(2*pi)*sigma^2) */
    for(int i=0;i<size;i++) stencil[i] = norm*(dx[i]*Gx[i]-dx[i+1]*Gx[i+1]);
}

inline
void fill_DXS2_stencil(int size, double stencil[], const double dx[], const double Gx[], const double DXS[], double theta_sigma)
{
    double sigma2=theta_sigma*theta_sigma;
    double norm=1./sqrt2pi/(sigma2*sigma2*theta_sigma); /* 1/(sqrt(2*pi)*sigma^5) */
    double recip=2./theta_sigma;
    for(int i=0;i<size;i++) {
        double d=dx[i];
        double d1=dx[i+1];
        stencil[i]=norm*(d*d*d*Gx[i]-d1*d1*d1*Gx[i+1]) - recip * DXS[i];
    }
}

inline
void fill_DXSX_stencil(int size, double stencil[], const double dx[], const double Gx[], const double DX[], double theta_sigma)
{
    double sigma2=theta_sigma*theta_sigma;
    double norm=1./sqrt2pi/(sigma2*sigma2); /* 1/(sqrt(2*pi)*sigma^4) */
    double recip=1./theta_sigma;
    for(int i=0;i<size;i++) {
        double d=dx[i];
        double d1=dx[i+1];
        stencil[i]=norm*(d*d*Gx[i]-d1*d1*Gx[i+1]) - recip * DX[i];
    }
}

inline
VecT make_gaussian_stencil(int size, double sigma)
{
    VecT stencil(2*size-1);
    fill_gaussian_stencil(size, stencil.memptr(), sigma);
    return stencil;
}

inline 
double poisson_log_likelihood(double model_val, double data_val)
{
    if(!std::isfinite(data_val)) return 0.; //This is a masked pixel do not include in likelihood
    if(model_val==0.) return 0.;//Probability here is below machine epsilon
    if(data_val==0.) return -model_val; //Skip multiplication by zero
    return data_val*log(model_val)-model_val-lgamma(data_val+1);
}

inline 
double relative_poisson_log_likelihood(double model_val, double data_val)
{
    if(!std::isfinite(data_val)) return 0.; //This is a masked pixel do not include in likelihood
    if(model_val==0.) return 0.;//Probability here is below machine epsilon
    if(data_val==0.) return -model_val; //Skip multiplication by zero
    return data_val*log(model_val)-model_val;
}

inline
double log_prior_beta_const(double beta)
{
    return -2*lgamma(beta)-lgamma(2*beta);
}

inline
double log_prior_beta2_const(double beta0, double beta1)
{
    return -lgamma(beta0)-lgamma(beta1)-lgamma(beta0+beta1);
}


inline
double log_prior_gamma_const(double kappa, double mean)
{
    return kappa*(log(kappa)-log(mean))-lgamma(kappa);
}

inline
double log_prior_pareto_const(double alpha, double min)
{
    return log(alpha) + alpha*log(min);
}

inline
double log_prior_normal_const(double sigma)
{
    return -log(sqrt(2*arma::datum::pi))-log(sigma);
}

inline
double rllh_beta_prior(double beta, double v, double max, double min)
{
    double d=max-min;
    return (beta-1)*(log((v-min)/d) - log((max-v)/d));
}

inline
double rllh_beta2_prior(double beta0, double beta1, double v, double max, double min)
{
    double d=max-min;
    return (beta1-1)*log((v-min)/d) - (beta0-1)*log((max-v)/d);
}

inline
double rllh_gamma_prior(double kappa, double mean, double v)
{
    return (kappa-1)*log(v) - kappa*v/mean;
}

inline
double rllh_pareto_prior(double alpha,  double v)
{
    return -(alpha+1)*log(v);
}

inline
double rllh_normal_prior(double mu,  double sigma, double v)
{
    double d=v-mu;
    return -(d*d)/(2*sigma);
}


inline
double beta_prior_grad(double beta, double v, double max, double min)
{
    double bm1=beta-1;
    return bm1/(v-min) - bm1/(max-v);
}

inline
double beta2_prior_grad(double beta0, double beta1, double v, double max, double min)
{
    return (beta1-1)/(v-min) - (beta0-1)/(max-v);
}

inline
double gamma_prior_grad(double kappa, double mean, double v)
{
    return (kappa-1)/v - kappa/mean;
}

inline
double pareto_prior_grad(double alpha,  double v)
{
    return -(alpha+1)/v;
}

inline
double normal_prior_grad(double mu,  double sigma, double v)
{
    return (mu-v)/sigma;
}


inline
double beta_prior_grad2(double beta, double v, double max, double min)
{
    double r1=v-min;
    double r2=max-v;
    double bm1=beta-1;
    return -bm1/(r1*r1) - bm1/(r2*r2);
}
inline
double beta2_prior_grad2(double beta0, double beta1, double v, double max, double min)
{
    double r1=v-min;
    double r2=max-v;
    return -(beta1-1)/(r1*r1) - (beta0-1)/(r2*r2);
}


inline
double gamma_prior_grad2(double kappa, double v)
{
    return -(kappa-1)/(v*v);
}

inline
double pareto_prior_grad2(double alpha,  double v)
{
    return (alpha+1)/(v*v);
}

inline
double normal_prior_grad2(double sigma)
{
    return -1./sigma;
}


} /* namespace mappel */

#endif /* _STENCIL_H */
