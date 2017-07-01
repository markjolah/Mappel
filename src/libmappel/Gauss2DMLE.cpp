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
    typename Gauss2DMLE::Stencil
    CGaussHeuristicEstimator<Gauss2DMLE>::compute_estimate(const ModelDataT &im, const ParamT &theta_init)
    {
        auto theta_est = model.make_param();
        if(model.size(0) == model.size(1) && model.psf_sigma(0) == model.psf_sigma(1)){ //only works for square images and iso-tropic psf
            arma::fvec4 ftheta_est;
            arma::fmat fim = arma::conv_to<arma::fmat>::from(im);  //Convert image to float from double
            cgauss::MLEInit(fim.memptr(),model.psf_sigma(0),model.size(0),ftheta_est.memptr());
            theta_est = arma::conv_to<arma::mat>::from(ftheta_est); //Convert theta back to double
            //Swap x/y and add .5 tp convert from CGauss to mappel coordinates
            float temp = theta_est(0)+.5;
            theta_est(0) = theta_est(1)+.5;
            theta_est(1) = temp;
        } else {
            throw MaximizerNotImplementedException("CGaussMLE");
        }
        return model.make_stencil(theta_est);
    }
    
    template<>
    typename Gauss2DMLE::Stencil
    CGaussMLE<Gauss2DMLE>::compute_estimate(const ModelDataT &im, const ParamT &theta_init)
    {
        auto theta_est=model.make_param();
        if(model.size(0) == model.size(1) && model.psf_sigma(0) == model.psf_sigma(1)){//only works for square images and iso-tropic psf
            arma::fvec ftheta_est(4);
            arma::fmat fim = arma::conv_to<arma::fmat>::from(im); //Convert image to float from double
            arma::fvec ftheta_init = arma::conv_to<arma::fvec>::from(theta_init);
            if(!ftheta_init.is_empty()){
                float temp = ftheta_init(0)-.5;
                ftheta_init(0) = ftheta_init(1)-.5;
                ftheta_init(1) = temp;
            }
            cgauss::MLEFit(fim.memptr(), model.psf_sigma(0), model.size(0), max_iterations, ftheta_init, ftheta_est.memptr());
            theta_est = arma::conv_to<arma::vec>::from(ftheta_est); //Convert theta back to double
            //Swap x/y and add .5 tp convert from CGauss to mappel coordinates
            float temp = theta_est(0)+.5;
            theta_est(0) = theta_est(1)+.5;
            theta_est(1) = temp;
        } else {
            throw MaximizerNotImplementedException("CGaussMLE");
        }
        return model.make_stencil(theta_est);
    }
    
    template<>
    typename Gauss2DMLE::Stencil
    CGaussMLE<Gauss2DMLE>::compute_estimate_debug(const ModelDataT &im, const ParamT &theta_init, ParamVecT &sequence)
    {
        auto theta_est=model.make_param();
        if(model.size(0) == model.size(1) && model.psf_sigma(0) == model.psf_sigma(1)){//only works for square images and iso-tropic psf
            arma::fvec ftheta_est(4);
            arma::fmat fim = arma::conv_to<arma::fmat>::from(im); //Convert image to float from double
            arma::fvec ftheta_init = arma::conv_to<arma::fvec>::from(theta_init);
            if(!ftheta_init.is_empty()){
                float temp = ftheta_init(0)-.5;
                ftheta_init(0) = ftheta_init(1)-.5;
                ftheta_init(1) = temp;
            }
            cgauss::MLEFit_debug(fim.memptr(), model.psf_sigma(0), model.size(0), max_iterations, ftheta_init, ftheta_est.memptr(),sequence);
            theta_est = arma::conv_to<arma::vec>::from(ftheta_est); //Convert theta back to double
            //Swap x/y and add .5 tp convert from CGauss to mappel coordinates
            {
                float temp = theta_est(0)+.5;
                theta_est(0) = theta_est(1)+.5;
                theta_est(1) = temp;
            }
            for(int n=0; n < static_cast<int>(sequence.n_cols); n++) {
                float temp = sequence(0,n)+.5;
                sequence(0,n) = sequence(1,n)+.5;
                sequence(1,n) = temp;
            }
        } else {
            throw MaximizerNotImplementedException("CGaussMLE");
        }
        return model.make_stencil(theta_est);
    }
    
} /* namespace mappel */
