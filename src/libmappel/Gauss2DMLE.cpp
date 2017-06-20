/** @file Gauss2DMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-25-2014
 * @brief The class definition and template Specializations for Gauss2DMLE
 */
#include "Gauss2DMLE.h"
#include "cGaussMLE/cGaussMLE.h"
#include "cGaussMLE/GaussLib.h"

/* Constant model estimator names: These are the estimator names we have defined for this class */

namespace mappel {
    
    /* Template Specializations */
    template<>
    Gauss2DMLE::Stencil
    CGaussHeuristicEstimator<Gauss2DMLE>::compute_estimate(const ModelDataT &im, const ParamT &theta_init)
    {
        auto theta_est=model.make_param();
        theta_est.zeros();
        if(model.size(0)==model.size(1) && model.psf_sigma(0)==model.psf_sigma(1)){ //only works for square images and iso-tropic psf
            float Nmax;
            arma::fvec4 ftheta_est;
            //Convert from double
            arma::fmat fim=arma::conv_to<arma::fmat>::from(im);
            //Compute
            CenterofMass2D(model.size(0), fim.memptr(), &ftheta_est[0], &ftheta_est[1]);
            GaussFMaxMin2D(model.size(0), model.psf_sigma(0), fim.memptr(), &Nmax, &ftheta_est[3]);
            ftheta_est[2]=std::max(0., (Nmax-ftheta_est[3])*2*arma::datum::pi*model.psf_sigma(0)*model.psf_sigma(0));
            //Back to double
            theta_est=arma::conv_to<arma::mat>::from(ftheta_est);
            //Swap x/y and add .5 tp convert from CGauss to mappel coordinates
            float temp=theta_est(0)+.5;
            theta_est(0)=theta_est(1)+.5;
            theta_est(1)=temp;
        } else {
            throw MaximizerNotImplementedException("CGaussMLE");
        }
        return model.make_stencil(theta_est);
    }
    
    
    template<>
    void
    CGaussMLE<Gauss2DMLE>::compute_estimate(const ModelDataT &im, const ParamT &theta_init, ParamT &theta, ParamT &crlb, double &llh)
    {
        if(model.size(0)==model.size(1) && model.psf_sigma(0)==model.psf_sigma(1)){//only works for square images and iso-tropic psf
            float fllh;
            arma::fvec4 fcrlb, ftheta;
            //Convert from double
            arma::fmat fim=arma::conv_to<arma::fmat>::from(im);
            //Compute
            MLEFit(fim.memptr(), model.psf_sigma(0), model.size(0), max_iterations, ftheta.memptr(), fcrlb.memptr(), &fllh);
            //Back to double
            theta=arma::conv_to<arma::vec>::from(ftheta);
            crlb=arma::conv_to<arma::vec>::from(fcrlb);
            //Swap x/y and add .5 tp convert from CGauss to mappel coordinates
            float temp=theta(0)+.5;
            theta(0)=theta(1)+.5;
            theta(1)=temp;
            llh=log_likelihood(model,im,model.make_stencil(theta));
        } else {
            throw MaximizerNotImplementedException("CGaussMLE");
        }
    }
    
} /* namespace mappel */
