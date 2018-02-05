/** @file Mappel_IFace.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for Mappel_IFace.
 */

#ifndef _MAPPEL_IFACE
#define _MAPPEL_IFACE

#include <sstream>
#include <iostream>

#include "MexIFace.h"
#include "PointEmitterModel.h"
#include "methods.h"

using namespace mexiface;
namespace mappel {

template<class Model>
class Mappel_IFace :  public MexIFace, public MexIFaceHandler<Model> {
public:
    using Stencil = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
//     using ImageT = typename Model::ImageT;
//     using ImageStackT = typename Model::ImageStackT;
    using ImagePixelT = typename Model::ImagePixelT;
    template<class T> using ImageShapeT = typename Model::template ImageShapeT<T>;
    template<class T> using ImageStackShapeT = typename Model::template ImageStackShapeT<T>;
    
    Mappel_IFace();

protected:    
    using MexIFaceHandler<Model>::obj;
    void objGetHyperparams();
    void objSetHyperparams();
    void objGetStats();
    
    void objBoundTheta();
    void objThetaInBounds();
    void objSamplePrior();
    
    void objModelImage();
    void objSimulateImage();
    
    void objLLH();
    void objModelGrad();
    void objModelHessian();
    void objModelHessianNegativeDefinite();
    
    void objModelObjective();
    void objModelObjectiveAposteriori();
    void objModelObjectiveLikelihood();
    void objModelObjectivePrior();

    void objEstimate();
    void objEstimateError();
    void objEstimatePosterior();
    
    void objExpectedInformation();
    void objCRLB();
    
    /* Degugging */    
    void objEstimateDebug();
    void objEstimatePosteriorDebug();
    void objModelObjectiveComponents();
    
    /* Static methods */
    void objCholesky();
    void objModifiedCholesky();
    void objCholeskySolve();
};



template<class Model>
class MappelFixedSigma_IFace : public Mappel_IFace<Model>
{
public:
    void objConstruct() override;
};

template<class Model>
class MappelVarSigma_IFace : public Mappel_IFace<Model>
{
public:
    void objConstruct() override;
};


template<class Model>
void MappelFixedSigma_IFace<Model>::objConstruct()
{
    this->checkNumArgs(1,2);
    auto size = MexIFace::getVec<typename Model::ImageSizeT>();
    auto min_sigma = MexIFace::getVec();
    this->outputHandle(new Model(size,min_sigma));
}

template<class Model>
void MappelVarSigma_IFace<Model>::objConstruct()
{
    this->checkNumArgs(1,3);
    auto size = MexIFace::getVec<Model::ImageSizeT>();
    auto min_sigma = MexIFace::getVec();
    auto max_sigma = MexIFace::getVec();
    this->outputHandle(new Model(size,min_sigma,max_sigma));
}


template<class Model>
Mappel_IFace<Model>::Mappel_IFace() 
{
    methodmap["getHyperparams"]=boost::bind(&Mappel_IFace::objGetHyperparams, this);
    methodmap["setHyperparams"]=boost::bind(&Mappel_IFace::objSetHyperparams, this);
    methodmap["getStats"]=boost::bind(&Mappel_IFace::objGetStats, this);
    
    methodmap["boundTheta"]=boost::bind(&Mappel_IFace::objBoundTheta, this);
    methodmap["thetaInBounds"]=boost::bind(&Mappel_IFace::objThetaInBounds, this);
    methodmap["samplePrior"]=boost::bind(&Mappel_IFace::objSamplePrior, this);
    
    methodmap["modelImage"]=boost::bind(&Mappel_IFace::objModelImage, this);
    methodmap["simulateImage"]=boost::bind(&Mappel_IFace::objSimulateImage, this);
    
    methodmap["LLH"]=boost::bind(&Mappel_IFace::objLLH, this);
    methodmap["modelGrad"]=boost::bind(&Mappel_IFace::objModelGrad, this);
    methodmap["modelHessian"]=boost::bind(&Mappel_IFace::objModelHessian, this);
    methodmap["objModelHessianNegativeDefinite"]=boost::bind(&Mappel_IFace::objModelHessianNegativeDefinite, this);

    methodmap["modelObjective"]=boost::bind(&Mappel_IFace::objModelObjective, this);
    methodmap["modelObjectiveAposteriori"]=boost::bind(&Mappel_IFace::objModelObjectiveAposteriori, this);
    methodmap["modelObjectiveLikelihood"]=boost::bind(&Mappel_IFace::objModelObjectiveLikelihood, this);
    methodmap["modelObjectivePrior"]=boost::bind(&Mappel_IFace::objModelObjectivePrior, this);

    methodmap["estimate"]=boost::bind(&Mappel_IFace::objEstimate, this);
    methodmap["estimateError"]=boost::bind(&Mappel_IFace::objEstimateError, this);
    methodmap["estimatePosterior"]=boost::bind(&Mappel_IFace::objEstimatePosterior, this);

    methodmap["expectedInformation"]=boost::bind(&Mappel_IFace::objExpectedInformation, this);
    methodmap["CRLB"]=boost::bind(&Mappel_IFace::objCRLB, this);

    /* Debug */
    methodmap["estimateDebug"]=boost::bind(&Mappel_IFace::objEstimateDebug, this);
    methodmap["estimatePosteriorDebug"]=boost::bind(&Mappel_IFace::objEstimatePosteriorDebug, this);
    methodmap["modelObjectiveComponents"]=boost::bind(&Mappel_IFace::objModelObjectiveComponents, this);

    /* Static debug */
    staticmethodmap["cholesky"]=boost::bind(&Mappel_IFace::objCholesky, this);
    staticmethodmap["modifiedCholesky"]=boost::bind(&Mappel_IFace::objModifiedCholesky, this);
    staticmethodmap["choleskySolve"]=boost::bind(&Mappel_IFace::objCholeskySolve, this);
}

template<class Model>
void Mappel_IFace<Model>::objGetHyperparams()
{
    // [hyperparams, hyperparams_desc] = obj.getHyperparams()
    //
    //(out) hyperparams: A (nHyperparams X 1) double of theta values
    //(out) hyperparams_dsec: A (nHyperparams X 1) cell array of strings with descriptions of variables
    checkMinNumArgs(1,0);
    checkMaxNumArgs(2,0);
    output(obj->get_hyperparams());
    if(nlhs>1) output(obj->get_hyperparams_desc());
}

template<class Model>
void Mappel_IFace<Model>::objSetHyperparams()
{
    // obj.setHyperparams(prior)
    //
    //(in) prior - double size:[nHyperparams, 1] 
    checkNumArgs(0,1);
    obj->set_hyperparams(getVec());
}

template<class Model>
void Mappel_IFace<Model>::objGetStats()
{
    // stats = obj.getStats();
    //
    // (out) stats - A Struct describing the class mapping names to values
    checkNumArgs(1,0);
    output(obj->get_stats());
}


template<class Model>
void Mappel_IFace<Model>::objBoundTheta()
{
    // bounded_theta = obj.boundTheta(theta)
    //
    // Truncates parameters values (theta) to ensure they are in-bounds
    //
    // [in] theta - double [nParams, n] stack of thetas to bound
    // [out] bounded_theta - double [nParams, n] stack of thetas truncated to be in bounds
    checkNumArgs(1,1);
    auto param = getMat();
    auto bounded = makeOutputArray(param.n_rows, param.n_cols);
    auto theta = obj->make_param();
    #pragma omp parallel for
    for(IdxT i=0; i<param.n_cols; i++) {
        theta = param.col(i);
        obj->bound_theta(theta);
        bounded.col(i) = theta;
    }
}

template<class Model>
void Mappel_IFace<Model>::objThetaInBounds()
{
    // in_bounds = obj.thetaInBounds(theta)
    //
    // Tests parameter values (theta) to ensure they are in-bounds
    //
    // [in] theta - double [nParams, n] stack of thetas to bound
    // [out] in_bounds - bool.  True if all theta are in bounds.
    checkNumArgs(1,1);
    auto param = getMat();
    bool ok = true;
    for(IdxT i=0; i<param.n_cols; i++) ok &= obj->theta_in_bounds(param.col(i));
    output(ok);
}

template<class Model>
void Mappel_IFace<Model>::objSamplePrior()
{
    // theta = obj.samplePrior(count)
    // [in] count: integer number of thetas to sample
    // [out] theta: sampled parameter values size:[nParams X count] 
    checkNumArgs(1,1);
    auto count = getAsInt<IdxT>();
    auto theta = makeOutputArray(obj->get_num_params(), count);
    methods::sample_prior_stack(*obj, theta);
}

template<class Model>
void Mappel_IFace<Model>::objModelImage()
{
    // image = obj.modelImage(theta)
    //
    // The model image is the emitter image without Poisson noise applied.  It represents
    // the expected (mean) photon count at each pixel according to the model and parameters. 
    // 
    // [in] theta: double size:[nParams, n] stack of theta values
    // [out] image: double size:[imsize... ,n] image stack
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto image_stack = obj->make_image_stack(theta_stack.n_cols);
    methods::model_image_stack(*obj, theta_stack, image_stack);
}

template<class Model>
void Mappel_IFace<Model>::objSimulateImage()
{
    // image = obj.simulateImage(theta, count)
    //
    // If theta is size:[nParams, 1] then count images with that theta are
    // simulated.  Default count is 1.  If theta is Size:[nParams, n] with n>1
    // then n images are simulated, each with a separate theta, and count is ignored.
    //
    // [in] theta: double size:[nParams, n] stack of theta values
    // [in] count: [optional] integer number of thetas to sample
    // [out] image: double size:[imsize... ,n] image stack
    checkNumArgs(1,2);
    auto theta_stack = getMat();
    IdxT count = theta_stack.n_cols;
    if (count==1) count = getAsInt<IdxT>();
    auto image_stack = obj->make_image_stack(count);
    methods::simulate_image_stack(*obj, theta_stack, image_stack);
}

template<class Model>
void Mappel_IFace<Model>::objLLH()
{
    // llh = obj.LLH(image, theta)
    //
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single LLH.  If there are N=1 images
    // and M>1 thetas, we return M LLHs of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N LLHs for each of the images given theta
    //
    // [in] image: double size:[imsize... ,n] image stack
    // [in] theta: double size:[nParams, n] stack of theta values
    // [out] llh: a (1 X max(M,N)) double of log_likelihoods
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto llh_stack = makeOutputArray(count);
    methods::log_likelihood_stack(*obj, image_stack, theta_stack, llh_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelGrad()
{
    // grad = obj.modelGrad(image, theta) - Compute the model gradiant.
    //
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single Grad.  If there are N=1 images
    // and M>1 thetas, we return M Grads of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N Grads for each of the images given theta
    //
    // [in] image: double size:[imsize... ,N] image stack
    // [in] theta: double size:[nParams, M] stack of theta values
    // [out] grad: double size:[nParams,max(M,N)] stack of corresponding gradient vectors
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto grad_stack = makeOutputArray(obj->get_num_params(), count);
    methods::model_grad_stack(*obj, image_stack, theta_stack, grad_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelHessian()
{
    // hess = obj.modelHessian(image, theta) - Compute the model hessian
    //
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single Hessian.  If there are N=1 images
    // and M>1 thetas, we return M Hessian of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N Hessians for each of the images given theta
    //
    // [in] image: double size:[imsize... ,N] image stack
    // [in] theta: double size:[nParams, M] stack of theta values
    // [out] hess: double size:[nParams,nParams,max(M,N)] stack of hessian matricies
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto hess_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(), count);
    methods::model_hessian_stack(*obj, image_stack, theta_stack, hess_stack);
    copy_Usym_mat_stack(hess_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelObjective()
{
    // [rllh,grad,hess,llh] = obj.modelObjective(image, theta) - 
    
    // A convenience function for objective based optimization.  
    // Works on a single image, theta and shares the
    // stencil to compute the LLH,Grad,Hessian as the 3 outputs.
    // This allows faster use with matlab optimization.
    
    // [in] image: an image, double size:[imsizeY,imsizeX] 
    // [in] theta: a parameter value size:[nParams,1] double of theta
    // [in] (optional) negate: boolean. true if objective should be negated, as is the case with
    //                 matlab minimization routines
    // [out] RLLH:  relative log likelihood scalar double 
    // [out] (optional) Grad: grad of log likelihood scalar double size:[nParams,1] 
    // [out] (optional) Hess: hessian of log likelihood double size:[nParams,nParams] 
    // [out] (optional) LLH: full log likelihood with constant terms, double size:[nParams,nParams] 
    checkMinNumArgs(1,3);
    checkMaxNumArgs(4,3);
    auto image = getNumeric<ImageShapeT,ImagePixelT>();
    auto theta = getVec();
    bool negate = getAsBool();
    double negate_scalar = negate ? -1 : 1;
    auto stencil = obj->make_stencil(theta);
    double rllh = negate_scalar*methods::objective::rllh(*obj, image, stencil);
    if(negate) rllh = -rllh;
    output(llh);
    if(nlhs==2) {
        //nargout=2 Output grad also
        output(negate_scalar*model::objective::grad(*obj, image, stencil));
    } else if(nlhs>=3) {
        //nargout=3 Output both grad and hess which can be computed simultaneously!
        auto grad = makeOutputArray(obj->get_num_params());
        auto hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        methods::objective::hessian(*obj, image, stencil, grad, hess);
        copy_Usym_mat(hess);
        grad *= negate_scalar;
        hess *= negate_scalar;
    }
    if(nlhs==4) {
        output(negate_scalar*methods::objective::llh(*obj, image, stencil)); //ouput optional full llh
    }
}

template<class Model>
void Mappel_IFace<Model>::objModelNegativeDefiniteHessian()
{
    // hess = obj.modelNegativeDefiniteHessian(image, theta)
    //
    // Compute the modified cholesky decomposition form of the hessian which
    // should be negative definite at the maximum.
    //
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single Hessian.  If there are N=1 images
    // and M>1 thetas, we return M Hessian of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N Hessians for each of the images given theta
    //
    // [in] image: double size:[imsize... ,N] image stack
    // [in] theta: double size:[nParams, M] stack of theta values
    // [out] hess: double size:[nParams,nParams,max(M,N)] stack of hessian matricies
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto hess_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(), count);
    model::model_negative_definite_hessian_stack(*obj, image_stack, theta_stack, hess_stack);
    copy_Usym_mat_stack(hess_stack);
}

template<class Model>
void Mappel_IFace<Model>::objCRLB()
{
    // crlb = obj.CRLB(theta) - Compute the Cramer-Rao Lower Bound at theta
    // 
    // The cramer-rao lower bound (CRLB) is the lower bound on the variance at 
    // theta for any unbiased estimator assuming a normal error distribution.  This is the best
    // possible case based on estimator theory.  The computation relies on the expectedInformation which
    // is independent of the images.
    // There is no guarantee the estimator meets the CRLB or that the errors are Gaussian or even symmetric about
    // the estimated value.  As the amount of data (photons) goes to infinity this should match up with
    // other methods of estimating the error profile.
    //
    // [in] theta: double size:[nParams, n] stack of theta values
    // [out] crlb: double size:[nParams, n] stack of  cramer-rao lower bound symmetric errors.
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto crlb_stack = makeOutputArray(obj->get_num_params(), theta_stack.n_cols);
    cr_lower_bound_stack(*obj, theta_stack, crlb_stack);
}

template<class Model>
void Mappel_IFace<Model>::objExpectedInformation()
{
    // fisherI = obj.expectedInformation(theta) - Compute the Expected (Fisher) Information matrix
    //    at theta
    // [in] theta: double size:[nParams, n] stack of theta values
    // [out] fisherI: double size:[nParams,nParms, n] stack if symmetric fisher information matricies at each theta 
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto fisherI_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(),theta_stack.n_cols);
    fisher_information_stack(*obj, theta_stack, fisherI_stack);
    copy_Usym_mat_stack(fisherI_stack);
}

template<class Model>
void Mappel_IFace<Model>::objEstimate()
{
    // [theta_est, rllh, obsI, stats] = obj.estimate(image, name, theta_init) 
    //
    // Use maximization algorithm to estimate parameter theta as a Maximum-likelihood or maximum a-posteriori
    // point estimate.
    //
    // Retuns the observedInformation matrix which can be used to estimate the error 
    // Also returns the relative_log_likelihood at each estimate.
    //
    // [in] image: double size:[imsize... ,n] image stack to run estimations on
    // [in] name: (optional) name for the optimization method. (default = 'Newton')  
    //      Valid names are in obj.EstimationMethods
    // [in] theta_init: (optional) double size:[nParams, n] initial theta values to use in optimization
    // [out] theta_est: double size:[nParams, n] Sequence of estimated theta values each theta is a column
    // [out] rllh: (optional) double size:[n] Relative log likelihood at each theta estimate.
    // [out] obsI: (optional) double size:[nParams,nParams,n] observed information is the negative hessian of log likelihood 
    //             at each theta as symmetric nParams X nParams matrix.
    // [out] stats: (optional) A 1x1 struct of estimator statistics.
    checkMinNumArgs(1,1);
    checkMaxNumArgs(5,3);
    //Get input
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    std::string name = getString();
    auto theta_init_stack = getMat();
    ParamVecT theta_init;
    int nimages = obj->size_image_stack(image_stack);
    //Make output
    auto theta_stack = makeOutputArray(obj->get_num_params(),nimages);
    auto crlb_stack = makeOutputArray(obj->get_num_params(),nimages);
    auto llh_stack = makeOutputArray(nimages);
    //Call method
    auto estimator = make_estimator(*obj, name);
    if(!estimator) {
        std::ostringstream out;
        out<<"Bad estimator name: "<<name;
        throw NotImplementedError(out.str());
    }
    estimator->estimate_stack(image_stack, theta_init_stack, theta_stack, crlb_stack, llh_stack);
    output(estimator->get_stats());
}

template<class Model>
void Mappel_IFace<Model>::objEstimateDebug()
{
    // [theta_est, rllh, obsI, stats, seqence, sequence_rllh] = obj.estimate(image, name, theta_init) 
    //
    //  Debug the estimation on a single image, returning the sequence of evaluated points and their rllh
    //
    // [in] image: double size:[imsize...] single image to debug  estimations on
    // [in] name:  name for the optimization method. (default = 'Newton')  
    //      Valid names are in obj.EstimationMethods
    // [in] theta_init: (optional) double size:[nParams, n] initial theta values to use in optimization
    // [out] theta_est: double size:[nParams, n] Sequence of estimated theta values each theta is a column
    // [out] rllh:  double size:[n] Relative log likelihood at each theta estimate.
    // [out] obsI:  double size:[nParams,nParams,n] observed information is the negative hessian of log likelihood 
    //             at each theta as symmetric [nParams, nParams] matrix.
    // [out] stats:  A 1x1 struct of estimator statistics.
    // [out] sequence:  double size:[nParams,n] sequence of evaluated thetas
    // [out] sequence_rllh:  double size:[nParams,n] sequence of evaluated theta relative_log_likelihood
    checkNumArgs(6,3);
    //Get input
    auto image = getNumeric<ImageShapeT,ImagePixelT>();
    auto name = getString();
    auto theta_init = getVec();
    //Make temporaries (don't know sequence length ahead of call)
    ParamT theta;
    ParamT crlb;
    double llh;
    ParamVecT sequence;
    arma::vec sequence_llh;
    ParamT theta_init_p;
    theta_init_p.zeros();
    if(!theta_init.is_empty()){
        theta_init_p = theta_init;
    }
    //Call method
    auto estimator = make_estimator(*obj, name);
    if(!estimator) {
        std::ostringstream out;
        out<<"Bad estimator name: "<<name;
        throw NotImplementedError(out.str());
    }
    estimator->estimate_debug(image, theta_init_p, theta, crlb, llh, sequence, sequence_llh);
    //Write output
    output(theta);
    output(crlb);
    output(llh);
    output(estimator->get_debug_stats()); //Get the debug stats which might include more info like backtrack_idxs
    output(sequence);
    output(sequence_llh);
}

template<class Model>
void Mappel_IFace<Model>::objEstimatePosterior()
{
    // [mean, cov, stats]=estimatePosterior(obj, image, max_iterations)
    // (in) image: a double (imsize X imsize X n) image stack
    // (in) Nsamples: A  number of samples to take
    // (in) theta_init: (optional) a (nParams x n) double of initial theta values to use for starting MCMC
    // (out) mean: a (nParams X n) double of estimated posterior mean values
    // (out) cov: a (nParams X nParams X n) estimate of the posterior covarience.
    checkNumArgs(2,3);
    //Get input
    auto ims = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto Nsamples = getAsInt<IdxT>();
    auto theta_init_stack = getMat();

    auto Nims = obj->size_image_stack(ims);
    auto Np = obj->get_num_params();//number of model parameters
    //Make output
    auto means = makeOutputArray(Np,Nims);
    auto covs = makeOutputArray(Np,Np,Nims);
    //Call method
    evaluate_posterior_stack(*obj, ims, theta_init_stack, Nsamples, means, covs);
}

template<class Model>
void Mappel_IFace<Model>::objEstimatePosteriorDebug()
{
    // [mean, cov, stats]=estimatePosterior(obj, image, max_iterations)
    // (in) image: a double (imsize X imsize ) image stack
    // (in) Nsamples: A  number of samples to take
    //  (in) theta_init: (optional) a (nParams x 1) double of initial theta values to use in optimization
    // (out) mean: a (nParams X 1) double of estimated posterior mean values
    // (out) cov: a (nParams X nParams X 1) estimate of the posterior covarience.
    // (out) sample: A (nParams X nsamples) array of thetas samples
    // (out) sample_llh: A (1 X nsmaples) array of relative log likelyhoods at each sample theta
    // (out) candidates: A (nParams X nsamples) array of candidate thetas
    // (out) candidate_llh: A (1 X nsmaples) array of relative log likelyhoods at each candidate theta
    checkNumArgs(6,3);
    //Get input
    auto im = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto Ns = getAsInt<IdxT>(); //number of samples
    auto theta_init = getVec();
    auto Np = obj->get_num_params();//number of model parameters
    //Make ouput
    auto mean = makeOutputArray(Np);
    auto cov = makeOutputArray(Np,Np);
    auto sample = makeOutputArray(Np,Ns);
    auto sample_llh = makeOutputArray(Ns);
    auto candidates = makeOutputArray(Np,Ns);
    auto candidate_llh = makeOutputArray(Ns);
    //Call method
    evaluate_posterior_debug(*obj,im,theta_init,Ns,mean,cov,sample,sample_llh,candidates,candidate_llh);
}



//Static debugging for cholesky

template<class Model>
void Mappel_IFace<Model>::objCholesky()
{
    checkNumArgs(3,1);
    auto A = getMat();
    if(!is_symmetric(A)) throw ArrayShapeError("Matrix is not symmetric");
    auto C = A;
    bool valid = cholesky(C);
    //Seperate d from C so C is unit lower triangular
    VecT d = C.diag();
    C.diag().fill(1.);
    output(valid);
    output(C);
    output(d);
}

template<class Model>
void Mappel_IFace<Model>::objModifiedCholesky()
{
    checkNumArgs(3,1);
    auto A = getMat();
    if(!is_symmetric(A)) throw ArrayShapeError("Matrix is not symmetric");
    auto C = A;
    bool modified = modified_cholesky(C);
    VecT d = C.diag();
    C.diag().fill(1.);
    output(modified);
    output(C);
    output(d);
}

template<class Model>
void Mappel_IFace<Model>::objCholeskySolve()
{
    checkNumArgs(2,2);
    auto A = getMat();
    auto b = getVec();
    if(!is_symmetric(A)) throw ArrayShapeError("Matrix is not symmetric");
    if(b.n_elem != A.n_rows) throw ArrayShapeError("Input sizes do not match");
    auto C = A;
    bool modified = modified_cholesky(C);
    auto x = cholesky_solve(C,b);
    output(modified);
    output(x);
}

} /* namespace mappel */

#endif /* _MAPPEL_IFACE */
