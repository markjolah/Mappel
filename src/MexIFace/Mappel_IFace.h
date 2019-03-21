/** @file Mappel_IFace.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for Mappel_IFace.
 */

#ifndef MAPPEL_MAPPEL_IFACE_H
#define MAPPEL_MAPPEL_IFACE_H

#include <sstream>
#include <iostream>
#include <functional>
#include <thread>
#include <omp.h>

#include "MexIFace/MexIFace.h"
#include "Mappel/PointEmitterModel.h"
#include "Mappel/model_methods.h"

using namespace mexiface;
using namespace mappel;

template<class Model>
class Mappel_IFace :  public MexIFace, public MexIFaceHandler<Model> {
public:
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ImagePixelT = typename Model::ImagePixelT;
    template<class T> using ImageShapeT = typename Model::template ImageShapeT<T>;
    template<class T> using ImageStackShapeT = typename Model::template ImageStackShapeT<T>;
    
    Mappel_IFace();

protected:    
    using MexIFaceHandler<Model>::obj;
    void objGetHyperparams();
    void objSetHyperparams();
    void objGetHyperparamNames();
    void objSetHyperparamNames();
    void objGetParamNames();
    void objSetParamNames();
    void objSetImageSize();
    void objGetBounds();
    void objSetBounds();

    void objBoundTheta();
    void objThetaInBounds();

    void objGetStats();
    void objSamplePrior();
    
    void objModelImage();
    void objSimulateImage();
    
    void objModelLLH();
    void objModelRLLH();
    void objModelGrad();
    void objModelHessian();
    
    void objModelObjective();
    void objModelObjectiveAPosteriori();
    void objModelObjectiveLikelihood();
    void objModelObjectivePrior();

    void objExpectedInformation();
    void objCRLB();

    void objEstimate();
    void objEstimateProfileLikelihood();
    void objEstimatePosterior();

    void objErrorBoundsObserved();
    void objErrorBoundsExpected();
    void objErrorBoundsProfileLikelihood();

    /* Degugging */    
    void objEstimateDebug();
    void objEstimatePosteriorDebug();
    void objErrorBoundsProfileLikelihoodDebug();
    void objModelObjectiveComponents();
    
    /* Static methods */
    void staticCholesky();
    void staticModifiedCholesky();
    void staticCholeskySolve();
    void staticNegativeDefiniteCholeskyApprox();
    void staticPositiveDefiniteCholeskyApprox();
};

template<class Model>
class MappelFixedSigma_IFace : public Mappel_IFace<Model>
{
public:
    using MexIFaceHandler<Model>::obj;
    MappelFixedSigma_IFace();
    void objConstruct() override;
    void objSetPSFSigma();
};

template<class Model>
class MappelVarSigma_IFace : public Mappel_IFace<Model>
{
public:
    using MexIFaceHandler<Model>::obj;
    MappelVarSigma_IFace();
    void objConstruct() override;
    void objSetMinSigma();
    void objSetMaxSigma();
};


template<class Model>
void MappelFixedSigma_IFace<Model>::objConstruct()
{
    this->checkNumArgs(1,2);
    auto size = MexIFace::getVec<typename Model::ImageCoordT>();
    auto min_sigma = MexIFace::getVec();
    this->outputHandle(new Model(size,min_sigma));
}

template<class Model>
void MappelVarSigma_IFace<Model>::objConstruct()
{
    this->checkNumArgs(1,3);
    auto size = MexIFace::getVec<typename Model::ImageCoordT>();
    auto min_sigma = MexIFace::getVec();
    auto max_sigma = MexIFace::getVec();
    this->outputHandle(new Model(size,min_sigma,max_sigma));
}


template<class Model>
Mappel_IFace<Model>::Mappel_IFace() 
{
    //This needs to be set for matlab to use all cores.
    omp_set_num_threads(std::thread::hardware_concurrency());

    //These are used to set properties.
    methodmap["getHyperparams"] = std::bind(&Mappel_IFace::objGetHyperparams, this);
    methodmap["setHyperparams"] = std::bind(&Mappel_IFace::objSetHyperparams, this);
    methodmap["getHyperparamNames"] = std::bind(&Mappel_IFace::objGetHyperparamNames, this);
    methodmap["setHyperparamNames"] = std::bind(&Mappel_IFace::objSetHyperparamNames, this);
    methodmap["getParamNames"] = std::bind(&Mappel_IFace::objGetParamNames, this);
    methodmap["setParamNames"] = std::bind(&Mappel_IFace::objSetParamNames, this);
    methodmap["setImageSize"] = std::bind(&Mappel_IFace::objSetImageSize, this);
    methodmap["getBounds"] = std::bind(&Mappel_IFace::objGetBounds, this);
    methodmap["setBounds"] = std::bind(&Mappel_IFace::objSetBounds, this);

    methodmap["boundTheta"] = std::bind(&Mappel_IFace::objBoundTheta, this);
    methodmap["thetaInBounds"] = std::bind(&Mappel_IFace::objThetaInBounds, this);

    methodmap["getStats"] = std::bind(&Mappel_IFace::objGetStats, this);

    //methodmap["setPriorType"] = std::bind(&Mappel_IFace::objSetPriorType, this);
    methodmap["samplePrior"] = std::bind(&Mappel_IFace::objSamplePrior, this);
    
    methodmap["modelImage"] = std::bind(&Mappel_IFace::objModelImage, this);
    methodmap["simulateImage"] = std::bind(&Mappel_IFace::objSimulateImage, this);
    
    methodmap["modelLLH"] = std::bind(&Mappel_IFace::objModelLLH, this);
    methodmap["modelRLLH"] = std::bind(&Mappel_IFace::objModelRLLH, this);
    methodmap["modelGrad"] = std::bind(&Mappel_IFace::objModelGrad, this);
    methodmap["modelHessian"] = std::bind(&Mappel_IFace::objModelHessian, this);

    methodmap["modelObjective"] = std::bind(&Mappel_IFace::objModelObjective, this);
    methodmap["modelObjectiveAPosteriori"] = std::bind(&Mappel_IFace::objModelObjectiveAPosteriori, this);
    methodmap["modelObjectiveLikelihood"] = std::bind(&Mappel_IFace::objModelObjectiveLikelihood, this);
    methodmap["modelObjectivePrior"] = std::bind(&Mappel_IFace::objModelObjectivePrior, this);

    methodmap["expectedInformation"] = std::bind(&Mappel_IFace::objExpectedInformation, this);
    methodmap["CRLB"] = std::bind(&Mappel_IFace::objCRLB, this);

    methodmap["estimate"] = std::bind(&Mappel_IFace::objEstimate, this);
    methodmap["estimateProfileLikelihood"] = std::bind(&Mappel_IFace::objEstimateProfileLikelihood, this);
    methodmap["estimatePosterior"] = std::bind(&Mappel_IFace::objEstimatePosterior, this);

    methodmap["errorBoundsObserved"] = std::bind(&Mappel_IFace::objErrorBoundsObserved, this);
    methodmap["errorBoundsExpected"] = std::bind(&Mappel_IFace::objErrorBoundsExpected, this);
    methodmap["errorBoundsProfileLikelihood"] = std::bind(&Mappel_IFace::objErrorBoundsProfileLikelihood, this);
    methodmap["errorBoundsProfileLikelihoodDebug"] = std::bind(&Mappel_IFace::objErrorBoundsProfileLikelihoodDebug, this);
    /* Debug */
    methodmap["estimateDebug"] = std::bind(&Mappel_IFace::objEstimateDebug, this);
    methodmap["estimatePosteriorDebug"] = std::bind(&Mappel_IFace::objEstimatePosteriorDebug, this);
    methodmap["modelObjectiveComponents"] = std::bind(&Mappel_IFace::objModelObjectiveComponents, this);

    /* Static debug */
    staticmethodmap["cholesky"] = std::bind(&Mappel_IFace::staticCholesky, this);
    staticmethodmap["modifiedCholesky"] = std::bind(&Mappel_IFace::staticModifiedCholesky, this);
    staticmethodmap["choleskySolve"] = std::bind(&Mappel_IFace::staticCholeskySolve, this);
    staticmethodmap["positiveDefiniteCholeskyApprox"] = std::bind(&Mappel_IFace::staticPositiveDefiniteCholeskyApprox, this);
    staticmethodmap["negativeDefiniteCholeskyApprox"] = std::bind(&Mappel_IFace::staticNegativeDefiniteCholeskyApprox, this);
}

template<class Model>
MappelFixedSigma_IFace<Model>::MappelFixedSigma_IFace() : Mappel_IFace<Model>()
{
    this->methodmap["setPSFSigma"] = std::bind(&MappelFixedSigma_IFace::objSetPSFSigma, this);
}

template<class Model>
MappelVarSigma_IFace<Model>::MappelVarSigma_IFace() : Mappel_IFace<Model>()
{
    this->methodmap["setMinSigma"] = std::bind(&MappelVarSigma_IFace::objSetMinSigma, this);
    this->methodmap["setMaxSigma"] = std::bind(&MappelVarSigma_IFace::objSetMaxSigma, this);
}


template<class Model>
void Mappel_IFace<Model>::objGetHyperparams()
{
    checkNumArgs(1,0);
    output(obj->get_hyperparams());
}

template<class Model>
void Mappel_IFace<Model>::objSetHyperparams()
{
    checkNumArgs(0,1);
    obj->set_hyperparams(getVec());
}

template<class Model>
void Mappel_IFace<Model>::objGetHyperparamNames()
{
    checkNumArgs(1,0);
    output(obj->get_hyperparam_names());
}

template<class Model>
void Mappel_IFace<Model>::objSetHyperparamNames()
{
    checkNumArgs(0,1);
    obj->set_hyperparam_names(getStringArray());
}

template<class Model>
void Mappel_IFace<Model>::objGetParamNames()
{
    checkNumArgs(1,0);
    output(obj->get_param_names());
}

template<class Model>
void Mappel_IFace<Model>::objSetParamNames()
{
    checkNumArgs(0,1);
    obj->set_param_names(getStringArray());
}

template<class Model>
void Mappel_IFace<Model>::objSetImageSize()
{
    checkNumArgs(0,1);
    obj->set_size(getVec<typename Model::ImageCoordT>());
}

template<class Model>
void Mappel_IFace<Model>::objGetBounds()
{
    checkNumArgs(2,0);
    output(obj->get_lbound());
    output(obj->get_ubound());
}

template<class Model>
void Mappel_IFace<Model>::objSetBounds()
{
    checkNumArgs(0,2);
    obj->set_lbound(getVec());
    obj->set_ubound(getVec());
}

template<class Model>
void Mappel_IFace<Model>::objBoundTheta()
{
    // bounded_theta = obj.boundTheta(theta)
    //
    // Truncates parameters values (theta) to ensure they are in-bounds
    //
    // (in) theta - double [NumParams, n] stack of thetas to bound
    // (out) bounded_theta - double [NumParams, n] stack of thetas truncated to be in bounds
    checkNumArgs(1,1);
    output(obj->bounded_theta_stack(getMat()));
}

template<class Model>
void Mappel_IFace<Model>::objThetaInBounds()
{
    // in_bounds = obj.thetaInBounds(theta)
    //
    // Tests parameter values (theta) to ensure they are in-bounds
    //
    // (in) theta - double [NumParams, n] stack of thetas to bound
    // (out) in_bounds - bool size:[n] vector indicating if each theta is in bounds
    checkNumArgs(1,1);
    output(obj->theta_stack_in_bounds(getMat()));
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
void Mappel_IFace<Model>::objSamplePrior()
{
    // theta = obj.samplePrior(count)
    // (in) count: integer number of thetas to sample
    // (out) theta: sampled parameter values size:[NumParams X count]
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
    // (in) theta: double size:[NumParams, n] stack of theta values
    // (out) image: double size:[imsize... ,n] image stack
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto image_stack = obj->make_image_stack(theta_stack.n_cols);
    methods::model_image_stack(*obj, theta_stack, image_stack);
    output(image_stack);
}

template<class Model>
void Mappel_IFace<Model>::objSimulateImage()
{
    // image = obj.simulateImage(theta, count)
    //
    // If theta is size:[NumParams, 1] then count images with that theta are
    // simulated.  Default count is 1.  If theta is Size:[NumParams, n] with n>1
    // then n images are simulated, each with a separate theta, and count is ignored.
    //
    // (in) theta: double size:[NumParams, n] stack of theta values
    // (in) count: [optional] integer number of thetas to sample
    // (out) image: double size:[imsize... ,n] image stack
    checkNumArgs(1,2);
    auto theta_stack = getMat();
    IdxT count = theta_stack.n_cols;
    if (count==1) count = getAsInt<IdxT>();
    auto image_stack = obj->make_image_stack(count);
    methods::simulate_image_stack(*obj, theta_stack, image_stack);
    output(image_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelLLH()
{
    // llh = obj.modelLLH(image, theta)
    //
    // This takes in a N images and M thetas.  If M=N=1,
    // then we return a single LLH.  If there are N=1 images
    // and M>1 thetas, we return M LLHs of the same image with each of
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N LLHs for each of the images given theta
    //
    // (in) image: double size:[imsize... ,n] image stack
    // (in) theta: double size:[NumParams, n] stack of theta values
    // (out) llh: a (1 X max(M,N)) double of log_likelihoods
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, static_cast<IdxT>(obj->get_size_image_stack(image_stack)));
    auto llh_stack = makeOutputArray(count);
    methods::objective::llh_stack(*obj, image_stack, theta_stack, llh_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelRLLH()
{
    // rllh = obj.modelRLLH(image, theta)
    //
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single RLLH.  If there are N=1 images
    // and M>1 thetas, we return M RLLHs of the same image with each of
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N RLLHs for each of the images given theta
    //
    // (in) image: double size:[imsize... ,n] image stack
    // (in) theta: double size:[NumParams, n] stack of theta values
    // (out) llh: a (1 X max(M,N)) double of relative log_likelihoods
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, static_cast<IdxT>(obj->get_size_image_stack(image_stack)));
    auto rllh_stack = makeOutputArray(count);
    methods::objective::rllh_stack(*obj, image_stack, theta_stack, rllh_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelGrad()
{
    // grad = obj.modelGrad(image, theta) - Compute the model gradient.
    //
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single Grad.  If there are N=1 images
    // and M>1 thetas, we return M Grads of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N Grads for each of the images given theta
    //
    // (in) image: double size:[imsize... ,N] image stack
    // (in) theta: double size:[NumParams, M] stack of theta values
    // (out) grad: double size:[NumParams,max(M,N)] stack of corresponding gradient vectors
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, static_cast<IdxT>(obj->get_size_image_stack(image_stack)));
    auto grad_stack = makeOutputArray(obj->get_num_params(), count);
    methods::objective::grad_stack(*obj, image_stack, theta_stack, grad_stack);
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
    // (in) image: double size:[imsize... ,N] image stack
    // (in) theta: double size:[NumParams, M] stack of theta values
    // (out) hess: double size:[NumParams,NumParams,max(M,N)] stack of hessian matrices
    checkNumArgs(1,2);
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, static_cast<IdxT>(obj->get_size_image_stack(image_stack)));
    auto hess_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(), count);
    methods::objective::hessian_stack(*obj, image_stack, theta_stack, hess_stack);
    copy_Usym_mat_stack(hess_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelObjective()
{
    // [rllh,grad,hess,definite_hess,llh] = obj.modelObjective(image, theta, negate) -
    //
    // Evaluate the model's objective function and its derivatives
    // Works on a single image, theta and shares the
    // stencil to compute the RLLH,Grad,Hessian as the 3 outputs, with optional outputs
    // of a (negative/positive) definite corrected hessian and the true LLH with constant terms
    //
    // (in) image: an image
    // (in) theta: a parameter value size:[NumParams,1] double of theta
    // (in) (optional) negate: boolean. true if objective should be negated, as is the case with
    //                 matlab minimization routines
    // (out) RLLH:  relative log likelihood scalar double
    // (out) (optional) Grad: grad of log likelihood scalar double size:[NumParams,1]
    // (out) (optional) Hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) definite_hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) LLH: full log likelihood with constant terms, double size:[NumParams,NumParams]
    checkMinNumArgs(1,3);
    checkMaxNumArgs(5,3);
    auto image = getNumeric<ImageShapeT,ImagePixelT>();
    auto theta = getVec();
    bool negate = (nrhs==2) ? false : getAsBool();
    double negate_scalar = negate ? -1 : 1;
    if(!obj->theta_in_bounds(theta)) {
        output(arma::datum::nan);
        auto grad = makeOutputArray(obj->get_num_params());
        auto hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        grad.fill(arma::datum::nan);
        hess.fill(arma::datum::nan);
        if(nlhs>=3) {
            auto def_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
            def_hess.fill(arma::datum::nan);
            if(nlhs>=4) output(arma::datum::nan);
        }
        return;
    }
    auto stencil = obj->make_stencil(theta);
    double rllh = negate_scalar*methods::objective::rllh(*obj, image, stencil);
    output(rllh);
    if(nlhs==2) {
        //Output grad also
        output( (negate_scalar * methods::objective::grad(*obj, image, stencil)).eval());
    } else if(nlhs>=3) {
        //Output both grad and hess which can be computed simultaneously!
        auto grad = makeOutputArray(obj->get_num_params());
        auto hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        methods::objective::hessian(*obj, image, stencil, grad, hess);
        copy_Usym_mat(hess);
        if(negate){
            grad = -grad;
            hess = -hess;
        }
        if(nlhs>=4) {
            auto definite_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
            definite_hess = hess;
            if(negate) cholesky_make_positive_definite(definite_hess);
            else cholesky_make_negative_definite(definite_hess);
        }
        if(nlhs==5) output(negate_scalar*methods::objective::llh(*obj, image, stencil)); //ouput optional full llh
    }
}


template<class Model>
void Mappel_IFace<Model>::objModelObjectiveAPosteriori()
{
    // [rllh,grad,hess,llh] = obj.modelObjectiveAPosteriori(image, theta, negate) -
    //
    // Evaluate the a posteriori objective irrespective of the model's MLE/MAP.
    // This is the log-likelihood plus the log-prior-likelihood.
    // Works on a single image, theta and shares the
    // stencil to compute the RLLH,Grad,Hessian as the 3 outputs, with optional outputs
    // of a (negative/positive) definite corrected hessian and the true LLH with constant terms
    //
    // (in) image: an image
    // (in) theta: a parameter value size:[NumParams,1] double of theta
    // (in) (optional) negate: boolean. true if objective should be negated, as is the case with
    //                 matlab minimization routines
    // (out) RLLH:  relative log likelihood scalar double
    // (out) (optional) Grad: grad of log likelihood scalar double size:[NumParams,1]
    // (out) (optional) Hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) definite_hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) LLH: full log likelihood with constant terms, double size:[NumParams,NumParams]
    checkMinNumArgs(1,3);
    checkMaxNumArgs(5,3);
    auto image = getNumeric<ImageShapeT,ImagePixelT>();
    auto theta = getVec();
    bool negate = (nrhs==2) ? false : getAsBool();
    if(!obj->theta_in_bounds(theta)) {
        output(arma::datum::nan);
        auto grad = makeOutputArray(obj->get_num_params());
        auto hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        grad.fill(arma::datum::nan);
        hess.fill(arma::datum::nan);
        if(nlhs>=3) {
            auto def_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
            def_hess.fill(arma::datum::nan);
            if(nlhs>=4) output(arma::datum::nan);
        }
        return;
    }
    auto stencil = obj->make_stencil(theta);

    double rllh;
    auto grad = obj->make_param();
    auto hess = obj->make_param_mat();
    methods::aposteriori_objective(*obj, image, stencil, rllh, grad, hess);
    copy_Usym_mat(hess);
    if(negate){
        rllh = -rllh;
        grad = -grad;
        hess = -hess;
    }
    output(rllh);
    if(nlhs>=2) output(grad);
    if(nlhs>=3) output(hess);
    if(nlhs>=4) {
        auto definite_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        definite_hess = hess;
        if(negate) cholesky_make_positive_definite(definite_hess);
        else cholesky_make_negative_definite(definite_hess);
    }
    if(nlhs>=5) output(methods::likelihood::llh(*obj, image, stencil) + obj->get_prior().llh(theta));
}

template<class Model>
void Mappel_IFace<Model>::objModelObjectiveLikelihood()
{
    // [rllh,grad,hess,llh] = obj.modelObjectiveLikelihood(image, theta, negate) -
    //
    // Evaluate the pure-likelihood based objective irrespective of the model's MLE/MAP.
    // Works on a single image, theta and shares the
    // stencil to compute the RLLH,Grad,Hessian as the 3 outputs, with optional outputs
    // of a (negative/positive) definite corrected hessian and the true LLH with constant terms
    //
    // (in) image: an image
    // (in) theta: a parameter value size:[NumParams,1] double of theta
    // (in) (optional) negate: boolean. true if objective should be negated, as is the case with
    //                 matlab minimization routines
    // (out) RLLH:  relative log likelihood scalar double
    // (out) (optional) Grad: grad of log likelihood scalar double size:[NumParams,1]
    // (out) (optional) Hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) definite_hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) LLH: full log likelihood with constant terms, double size:[NumParams,NumParams]
    checkMinNumArgs(1,3);
    checkMaxNumArgs(5,3);
    auto image = getNumeric<ImageShapeT,ImagePixelT>();
    auto theta = getVec();
    bool negate = (nrhs==2) ? false : getAsBool();
    if(!obj->theta_in_bounds(theta)) {
        output(arma::datum::nan);
        auto grad = makeOutputArray(obj->get_num_params());
        auto hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        grad.fill(arma::datum::nan);
        hess.fill(arma::datum::nan);
        if(nlhs>=3) {
            auto def_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
            def_hess.fill(arma::datum::nan);
            if(nlhs>=4) output(arma::datum::nan);
        }
        return;
    }
    auto stencil = obj->make_stencil(theta);

    double rllh;
    auto grad = obj->make_param();
    auto hess = obj->make_param_mat();
    methods::likelihood_objective(*obj, image, stencil, rllh, grad, hess);
    copy_Usym_mat(hess);
    if(negate){
        rllh = -rllh;
        grad = -grad;
        hess = -hess;
    }
    output(rllh);
    if(nlhs>=2) output(grad);
    if(nlhs>=3) output(hess);
    if(nlhs>=4) {
        auto definite_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        definite_hess = hess;
        if(negate) {
            cholesky_make_positive_definite(definite_hess);
        } else {
            cholesky_make_negative_definite(definite_hess);
        }
    }
    if(nlhs>=5) output(methods::likelihood::llh(*obj, image, stencil));
}

template<class Model>
void Mappel_IFace<Model>::objModelObjectivePrior()
{
    // [rllh,grad,hess,llh] = obj.modelObjectivePrio(image, theta, negate) -
    //
    // Evaluate the pure-prior likelihood based objective irrespective of the model's MLE/MAP.
    // Works on a single image, theta and shares the
    // stencil to compute the RLLH,Grad,Hessian as the 3 outputs, with optional outputs
    // of a (negative/positive) definite corrected hessian and the true LLH with constant terms
    //
    // (in) theta: a parameter value size:[NumParams,1] double of theta
    // (in) (optional) negate: boolean. true if objective should be negated, as is the case with
    //                 matlab minimization routines
    // (out) RLLH:  relative log likelihood scalar double
    // (out) (optional) Grad: grad of log likelihood scalar double size:[NumParams,1]
    // (out) (optional) Hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) definite_hess: hessian of log likelihood double size:[NumParams,NumParams]
    // (out) (optional) LLH: full log likelihood with constant terms, double size:[NumParams,NumParams]
    checkMinNumArgs(1,2);
    checkMaxNumArgs(5,2);
    auto theta = getVec();
    bool negate = (nrhs==2) ? false : getAsBool();
    double negate_scalar = negate ? -1 : 1;

    double rllh;
    auto grad = obj->make_param();
    auto hess = obj->make_param_mat();
    methods::prior_objective(*obj, theta, rllh, grad, hess);
    copy_Usym_mat(hess);
    rllh *= negate_scalar;
    grad *= negate_scalar;
    hess *= negate_scalar;
    output(rllh);
    if(nlhs>=2) output(grad);
    if(nlhs>=3) output(hess);
    if(nlhs>=4) {
        auto definite_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        definite_hess = hess;
        definite_hess = hess;
        if(negate) cholesky_make_positive_definite(definite_hess);
        else cholesky_make_negative_definite(definite_hess);
    }
    if(nlhs>=5) output(obj->get_prior().llh(theta));
}

template<class Model>
void Mappel_IFace<Model>::objExpectedInformation()
{
    // fisherI = obj.expectedInformation(theta) - Compute the Expected (Fisher) Information matrix
    //    at theta
    // (in) theta: double size:[NumParams, n] stack of theta values
    // (out) fisherI: double size:[NumParams,nParms, n] stack if symmetric fisher information matrices at each theta
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto fisherI_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(),theta_stack.n_cols);
    methods::expected_information_stack(*obj, theta_stack, fisherI_stack);
    copy_Usym_mat_stack(fisherI_stack);
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
    // (in) theta: double size:[NumParams, n] stack of theta values
    // (out) crlb: double size:[NumParams, n] stack of  cramer-rao lower bound symmetric errors.
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto crlb_stack = makeOutputArray(obj->get_num_params(), theta_stack.n_cols);
    methods::cr_lower_bound_stack(*obj, theta_stack, crlb_stack);
}


template<class Model>
void Mappel_IFace<Model>::objEstimate()
{
    // [theta_est, rllh, obsI, stats] = obj.estimate(image, name, theta_init) 
    //
    // Use maximization algorithm to estimate parameter theta as a Maximum-likelihood or maximum a-posteriori
    // point estimate.
    //
    // Returns the observedInformation matrix which can be used to estimate the error
    // Also returns the relative_log_likelihood at each estimate.
    //
    // (in) image: double size:[imsize... ,n] image stack to run estimations on
    // (in) method_name: (optional) name for the optimization method. (default = 'Newton')
    //      Valid names are in obj.EstimationMethods
    // (in) theta_init: (optional) double size:[NumParams, n] initial theta values to use in optimization
    // (out) theta_est: double size:[NumParams, n] Sequence of estimated theta values each theta is a column
    // (out) rllh: (optional) double size:[n] Relative log likelihood at each theta estimate.
    // (out) obsI: (optional) double size:[NumParams,NumParams,n] observed information is the negative hessian of log likelihood
    //             at each theta as symmetric NumParams X NumParams matrix.
    // (out) stats: (optional) A 1x1 struct of estimator statistics.
    checkMinNumArgs(1,3);
    checkMaxNumArgs(4,3);
    //Get input
    auto image_stack = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto method_name = getString();
    auto theta_init_stack = getMat();
    estimator::MLEDataStack mle;
    mle.Ndata = obj->get_size_image_stack(image_stack);
    auto Np = obj->get_num_params();
    mle.theta = makeOutputArray(Np,mle.Ndata);
    mle.rllh = makeOutputArray(mle.Ndata);
    mle.obsI = makeOutputArray(Np,Np,mle.Ndata);

    if(nlhs==4){
        StatsT stats;
        methods::estimate_max_stack(*obj, image_stack,method_name,theta_init_stack, mle, stats);
        output(stats);
    }  else {
        methods::estimate_max_stack(*obj, image_stack,method_name,theta_init_stack, mle);
    }
    if(nlhs>=3) copy_Usym_mat_stack(mle.obsI);
}

template<class Model>
void Mappel_IFace<Model>::objEstimateDebug()
{
    // [theta_est, rllh, obsI, stats, seqence, sequence_rllh] = obj.estimate(image, name, theta_init) 
    //
    //  Debug the estimation on a single image, returning the sequence of evaluated points and their rllh
    //
    // (in) image: double size:[imsize...] single image to debug  estimations on
    // (in) method_name:  name for the optimization method.  Valid names are in obj.EstimationMethods
    // (in) theta_init: double size:[NumParams, n] initial theta values to use in optimization
    // (out) theta_est: double size:[NumParams, n] Sequence of estimated theta values each theta is a column
    // (out) rllh:  double size:[n] Relative log likelihood at each theta estimate.
    // (out) obsI:  double size:[NumParams,NumParams,n] observed information is the negative hessian of log likelihood
    //             at each theta as symmetric [NumParams, NumParams] matrix.
    // (out) sequence:  double size:[NumParams,n] sequence of evaluated thetas
    // (out) sequence_rllh:  double size:[NumParams,n] sequence of evaluated theta relative_log_likelihood
    // (out) stats:  A 1x1 struct of estimator statistics.
    checkMinNumArgs(1,3);
    checkMaxNumArgs(6,3);
    //Get input
    auto image = getNumeric<ImageShapeT,ImagePixelT>();
    auto method_name = getString();
    auto theta_init = getVec();
    estimator::MLEDebugData mle;
    StatsT stats;
    auto Np = obj->get_num_params();
    mle.theta = makeOutputArray(Np);
    methods::estimate_max_debug(*obj, image, method_name, theta_init, mle, stats);

    if(nlhs>=2) output(mle.rllh);
    if(nlhs>=3) {copy_Usym_mat(mle.obsI); output(mle.obsI); }
    if(nlhs>=4) output(mle.sequence);
    if(nlhs>=5) output(mle.sequence_rllh);
    if(nlhs>=6) output(stats); //Get the debug stats which might include more info like backtrack_idxs
}

template<class Model>
void Mappel_IFace<Model>::objEstimateProfileLikelihood()
{
    // [profile_likelihood, profile_parameters,stats]  = obj.estimateProfileLikelihood(image, fixed_parameters, fixed_values, estimator_algorithm, theta_init)
    //
    // Compute the profile log-likelihood for a single image and single parameter, over a range of values.  For each value, the parameter
    // of interest is fixed and the other parameters are optimized with the estimator_algorithm.
    // Values will be computed in parallel with OpenMP.
    //
    // (in) image: a single images
    // (in) fixed_idxs: uint64 [Nfixed,1] List of fixed indexs.  At least one parameter must be fixed and at least one must be free.
    // (in) fixed_values: size:[NumFixedParams,N], a vector of N values for each of the fixed parameters at which to maximize the other (free) parameters at.
    // (in) estimator_algorithm: (optional) name for the optimization method. (default = 'TrustRegion') [see: obj.EstimationMethods]
    // (in) theta_init: (optional) Initial theta guesses size:[NumParams,n]. [default: [] ] Empty array to force auto estimation.
    //                  If only a single parameter [NumParams,1] is given, each profile estimation will use this single theta_init.
    //                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
    //                   should be auto estimated.  The values vector will be automatically substituted for any fixed parameters.  Therefore they
    //                   do not need to be set in theta_init.
    // (out) profile_likelihood: size:[1,N] profile likelihood for the fixed parameter(s) at each value.,
    // (out) profile_parameters: size:[NumParams,N] parameters that achieve the profile likelihood maximum at each value.
    // (out) stats: (optional)  Estimator stats dictionary.
    checkMinNumArgs(1,5);
    checkMaxNumArgs(3,5);
    auto ims = getNumeric<ImageShapeT,ImagePixelT>();
    estimator::ProfileLikelihoodData prof;
    prof.fixed_idxs = getVec<IdxT>();
    prof.fixed_values = getMat();
    auto estimator_algorithm = getString();
    auto theta_init = getMat();
    StatsT stats;
    methods::estimate_profile_likelihood_stack(*obj, ims, estimator_algorithm, theta_init,  prof,stats);
    if(nlhs>=1) output(prof.profile_likelihood);
    if(nlhs>=2) output(prof.profile_parameters);
    if(nlhs>=3) output(stats);
}

template<class Model>
void Mappel_IFace<Model>::objEstimatePosterior()
{
    // [posterior_mean, credible_lb, credible_ub, posterior_cov, mcmc_samples]
    //      = obj.estimatePosterior(image, theta_init, confidence, num_samples, burnin, thin)
    //
    // Use MCMC sampling to sample from the posterior distribution and estimate the posterior_mean, credible interval upper and lower bounds for each parameter
    // and posterior covariance. Optionally returns mcmc-posterior sample for further post-processing.
    //
    // MCMC sampling can be controlled with the optional num_samples, burnin, and thin arguments.
    //
    // Confidence sets the confidence interval with for the credible interval lb and ub.  These are per parameter credible intervals.
    //
    // (in) image: a stack of n images to estimate
    // (in) theta_init: (optional) Initial theta guesses size:[NumParams,1]. [default: [] ] Use empty array to force auto estimation.
    //                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
    //                   should be auto estimated, valid parameter values will be kept even if invalid ones require initialization.
    // (in) confidence: (optional) desired confidence to estimate credible interval at.  Given as 0<confidence<1. [default=obj.DefaultConfidenceLevel]
    // (in) num_samples: (optional) Number of (post-filtering) posterior samples to acquire. [default=obj.DefaultMCMCNumSamples]
    // (in) burnin: (optional) Number of samples to throw away (burn-in) on initialization [default=obj.DefaultMCMCBurnin]
    // (in) thin: (optional) Keep every # samples. Value of 0 indicates use the model default. This is suggested.
    //                       When thin=1 there is no thinning.  This is also a good option if rejections are rare. [default=obj.DefaultMCMCThin]
    // (out) posterior_mean: size:[NumParams,n] posterior mean for each image
    // (out) credible_lb: (optional) size:[NumParams,n] posterior credible interval lower bounds for each parameter for each image
    // (out) credible_ub: (optional) size:[NumParams,n] posterior credible interval upper bounds for each parameter for each image
    // (out) posterior_cov: (optional) size:[NumParams,NumParams,n] posterior covariance matrices for each image
    // (out) mcmc_samples: (optional) size:[NumParams,max_samples,n] complete sequence of posterior samples generated by MCMC for each image
    // (out) mcmc_sample_llh: (optional) size:[max_samples,n] relative log likelihood of sequence of posterior samples generated by MCMC each column corresponds to an image.
    checkMinNumArgs(1,6);
    checkMaxNumArgs(6,6);
    //Get input
    auto ims = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_init_stack = getMat(); //Matlab iface code ensures we get proper shaped array even if user passes empty
    mcmc::MCMCDataStack mcmc;
    mcmc.Ndata = obj->get_size_image_stack(ims);
    mcmc.confidence = getAsFloat();
    mcmc.Nsample = getAsInt<IdxT>();
    mcmc.Nburnin = getAsInt<IdxT>();
    mcmc.thin = getAsInt<IdxT>();

    methods::estimate_posterior_stack(*obj, ims, theta_init_stack, mcmc);
    if(nlhs>=1) output(mcmc.sample_mean);
    if(nlhs>=2) output(mcmc.credible_lb);
    if(nlhs>=3) output(mcmc.credible_ub);
    if(nlhs>=4) output(mcmc.sample_cov);
    if(nlhs>=5) output(mcmc.sample);
    if(nlhs>=6) output(mcmc.sample_rllh);
}

template<class Model>
void Mappel_IFace<Model>::objEstimatePosteriorDebug()
{
    // [sample, sample_llh, candidates, candidate_llh] = obj.estimatePosteriorDebug(image, theta_init, num_samples)
    //
    // Debugging routine.  Works on a single image.  Get out the exact MCMC sample sequence, as well as the candidate sequence.
    // Does not do burnin or thinning.
    //
    // (in) image: a single image to estimate
    // (in) theta_init: (optional) Initial theta guesses size:[NumParams,1]. [default: [] ] Empty array to force auto estimation.
    //                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
    //                   should be auto estimated.
    // (in) num_samples: (optional) Number of (post-filtering) posterior samples to acquire. [default=obj.DefaultMCMCNumSamples]
    // (out) sample: A size:[NumParams,num_samples] array of thetas samples
    // (out) sample_llh: A size:[1,num_samples] array of log likelihoods at each sample theta
    // (out) candidates: (optional) size:[NumParams, num_samples] array of candidate thetas
    // (out) candidate_llh: (optional) A size:[1, num_samples] array of log likelihoods at each candidate theta
    checkMinNumArgs(1,3);
    checkMaxNumArgs(4,3);
    //Get input
    auto im = getNumeric<ImageShapeT,ImagePixelT>(); //single image
    mcmc::MCMCDebugData mcmc;
    auto theta_init = getVec();
    mcmc.Nsample = getAsInt<IdxT>();
    //Call method
    methods::estimate_posterior_debug(*obj,im,theta_init,mcmc);
    output(mcmc.sample);
    output(mcmc.sample_rllh);
    output(mcmc.candidate);
    output(mcmc.candidate_rllh);
}


template<class Model>
void Mappel_IFace<Model>::objErrorBoundsObserved()
{
    // [observed_lb, observed_ub] = obj.errorBoundsObserved(image, theta_mle, confidence, obsI)
    //
    // Compute the error bounds using the observed information at the MLE estimate theta_mle.
    //
    // (in) image: a single image or a stack of n images
    // (in) theta_mle: the MLE estimated theta size:[NumParams,n]
    // (in) confidence: (optional) desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
    // (in) obsI: (optional) observed information, at each theta_mle, as returned by obj.estimate size:[NumParams,NumParams,n].
    //                       Must be recomputed if not provided.
    // (out) observed_lb: the observed-information-based confidence interval lower bound for parameters, size:[NumParams,n]
    // (out) observed_ub: the observed-information-based confidence interval upper bound for parameters, size:[NumParams,n]
    checkMinNumArgs(2,3);
    checkMaxNumArgs(2,4);
    auto ims = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto thetas = getMat();
    double confidence = getAsFloat();
    auto Np = obj->get_num_params();
    auto Nims = obj->get_size_image_stack(ims);
    CubeT obsI;
    if(nrhs==4) {
        obsI = getCube();
    } else {
        obsI.set_size(Np,Np,Nims);
        methods::objective::hessian_stack(*obj,ims,thetas,obsI);
        obsI = -obsI;
    }
    auto observed_lb = makeOutputArray(Np,Nims);
    auto observed_ub = makeOutputArray(Np,Nims);
    methods::error_bounds_observed_stack(*obj, thetas, obsI, confidence, observed_lb, observed_ub);
}

template<class Model>
void Mappel_IFace<Model>::objErrorBoundsExpected()
{
    // [expected_lb, expected_ub] = obj.errorBoundsExpected(theta_mle, confidence)
    //
    // Compute the error bounds using the expected (Fisher) information at the MLE estimate.  This is independent of the image, assuming
    // Gaussian error with variance given by CRLB.
    //
    // (in) theta_mle: the theta MLE's to estimate error bounds for. size:[NumParams,N]
    // (in) confidence: (optional)  desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
    // (out) expected_lb: the expected-information-based confidence interval lower bound for parameters for each theta, size:[NumParams,N]
    // (out) expected_ub: the expected-information-based confidence interval upper bound for parameters for each theta, size:[NumParams,N]
    checkNumArgs(2,2);
    auto thetas = getMat();
    auto Nthetas = thetas.n_cols;
    auto Np = obj->get_num_params();//number of model parameters
    double confidence = getAsFloat();
    auto expected_lb = makeOutputArray(Np,Nthetas);
    auto expected_ub = makeOutputArray(Np,Nthetas);
    methods::error_bounds_expected_stack(*obj,thetas,confidence,expected_lb,expected_ub);
}

template<class Model>
void Mappel_IFace<Model>::objErrorBoundsProfileLikelihood()
{
    // [profile_lb, profile_ub, profile_points_lb, profile_points_ub, profile_points_lb_rllh, profile_points_ub_rllh, stats]
    //    = obj.errorBoundsProfileLikelihood(images, theta_mle, confidence, theta_mle_rllh, obsI, estimate_parameters)
    //
    // Compute the profile log-likelihood bounds for a stack of images, estimating upper and lower bounds for each requested parameter.
    // Uses the Venzon and Moolgavkar (1988) algorithm, implemented in OpenMP.
    //
    // (in) images: a stack of N images
    // (in) theta_mle: a stack of N theta_mle estimates corresponding to the image size
    // (in) confidence: (optional) desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
    // (in) theta_mle_rllh: [optional] size:[N] relative-log-likelihood at each image's theta_mle.  Otherwise it must be re-computed.
    // (in) obsI: [optional] observed information, at each theta_mle, as returned by obj.estimate size:[NumParams,NumParams,n].
    //                      Must be recomputed if not provided.
    // (in) estimated_idxs: [optional] uint64 indexes of estimated parameters. Empty to compute all parameters.
    // (out) profile_lb: size [NumParams,N] lower bounds for each parameter to be estimated. NaN if parameter was not estimated
    // (out) profile_ub: size [NumParams,N] upper bounds for each parameter to be estimated. NaN if parameter was not estimated
    // (out) profile_points_lb: [optional] size[NumParams,2,NumParams,N] Profile maxima thetas at which
    //           profile bounds were obtained.  Each [NumParams,2,NumParams] slice are the thetas found defining the
    //           the lower and upper bound for each parameter in sequence as the 3-rd dimension.
    //           The 4-th dimension is used if the profile is run on multiple images.  These can
    //           be useful to test for the quality of the estimated points.
    // (out) profile_points_ub:
    // (out) profile_points_lb_rllh: [optional] size [2,NumParams,N], rllh at each returned profile_points_lb.
    // (out) profile_points_ub_rllh: [optional] size [2,NumParams,N], rllh at each returned profile_points_ub.
    // (out) stats: struct of fitting statistics.
    checkMinNumArgs(2,3);
    checkMaxNumArgs(7,6);
    auto ims = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto Ndata = obj->get_size_image_stack(ims);
    if(Ndata==1) {
        estimator::ProfileBoundsData prof;
        prof.mle.theta = getVec();
        prof.confidence = getAsFloat();
        if(nrhs>=4) prof.mle.rllh = getAsFloat();
        if(nrhs>=5) prof.mle.obsI = getMat();
        if(nrhs==6) prof.estimated_idxs = getVec<IdxT>();
        StatsT stats;
        methods::error_bounds_profile_likelihood_parallel(*obj, ims, prof, stats);
        output(prof.profile_lb);
        output(prof.profile_ub);
        if(nlhs>=3) output(prof.profile_points_lb);
        if(nlhs>=4) output(prof.profile_points_ub);
        if(nlhs>=5) output(prof.profile_points_lb_rllh);
        if(nlhs>=6) output(prof.profile_points_ub_rllh);
        if(nlhs==7) output(stats);
    } else {
        estimator::ProfileBoundsDataStack prof;
        prof.Ndata = obj->get_size_image_stack(ims);
        prof.mle.theta = getMat();
        prof.confidence = getAsFloat();
        if(nrhs>=4) prof.mle.rllh = getVec();
        if(nrhs>=5) prof.mle.obsI = getCube();
        if(nrhs==6) prof.estimated_idxs = getVec<IdxT>();
        StatsT stats;
        methods::error_bounds_profile_likelihood_stack(*obj, ims, prof, stats);
        output(prof.profile_lb);
        output(prof.profile_ub);
        if(nlhs>=3) output(prof.profile_points_lb);
        if(nlhs>=4) output(prof.profile_points_ub);
        if(nlhs>=5) output(prof.profile_points_lb_rllh);
        if(nlhs>=6) output(prof.profile_points_ub_rllh);
        if(nlhs==7) output(stats);
    }

}

template<class Model>
void Mappel_IFace<Model>::objErrorBoundsProfileLikelihoodDebug()
{
    // [profile_lb, profile_ub, seq_lb, seq_ub, seq_lb_rllh, seq_ub_rllh, stats]
    //      = obj.errorBoundsProfileLikelihoodDebug(image, theta_mle, theta_mle_rllh, obsI, estimate_parameter, llh_delta)
    //
    // [DEBUGGING]
    // Compute the profile log-likelihood bounds for a single images , estimating upper and lower bounds for each requested  parameter.
    // Uses the Venzon and Moolgavkar (1988) algorithm.
    //
    // (in) image: a single images
    // (in) theta_mle:  theta ML estimate
    // (in) theta_mle_rllh:  scalar relative-log-likelihood at theta_mle.
    // (in) obsI: Observed fisher information matrix at theta_mle
    // (in) estimate_parameter_idx: integer index of parameter to estimate size:[NumParams]
    // (in) llh_delta:  Negative number, indicating LLH change from maximum at the profile likelihood boundaries.
    //                  [default: confidence=0.95; llh_delta = -chi2inv(confidence,1)/2;]
    // (out) profile_lb:  scalar lower bound for parameter
    // (out) profile_ub:  scalar upper bound for parameter
    // (out) seq_lb: size:[NumParams,Nseq_lb]  Sequence of Nseq_lb points resulting from VM algorithm for lower bound estimate
    // (out) seq_ub: size:[NumParams,Nseq_ub]  Sequence of Nseq_ub points resulting from VM algorithm for upper bound estimate
    // (out) seq_lb_rllh: size:[Nseq_lb]  Sequence of RLLH at each of the seq_lb points
    // (out) seq_ub_rllh: size:[Nseq_ub]  Sequence of RLLH at each of the seq_ub points
    // (out) stats: struct of fitting statistics.
    checkMinNumArgs(2,6);
    checkMaxNumArgs(7,6);
    auto im = getNumeric<ImageStackShapeT,ImagePixelT>();
    estimator::ProfileBoundsDebugData prof;
    prof.mle.theta = getMat();
    prof.mle.rllh = getAsFloat();
    prof.mle.obsI = getMat();
    prof.estimated_idx = getAsInt<IdxT>();
    prof.target_rllh_delta = getAsFloat();
    StatsT stats;
    if(!std::isfinite(prof.mle.rllh) || prof.mle.obsI.is_empty()) {
        auto stencil = obj->make_stencil(prof.mle.theta);
        prof.mle.rllh = methods::objective::rllh(*obj,im,stencil);
        prof.mle.obsI = methods::observed_information(*obj,im,stencil);
    }
    methods::error_bounds_profile_likelihood_debug(*obj, im, prof, stats);
    output(prof.profile_lb);
    output(prof.profile_ub);
    if(nlhs>=3) output(prof.sequence_lb);
    if(nlhs>=4) output(prof.sequence_ub);
    if(nlhs>=5) output(prof.sequence_lb_rllh);
    if(nlhs>=6) output(prof.sequence_ub_rllh);
    if(nlhs==7) output(stats);
}

template<class Model>
void Mappel_IFace<Model>::objModelObjectiveComponents()
{
    // [rllh_components, grad_components, hess_components] = obj.modelObjectiveComponents(image, theta)
    //
    // [DEBUGGING]
    //  Component-wise break down of model objective into individual contributions from pixels and model components.
    //  Each pixel that is not NAN (as well as the prior for MAP models) will contribute linearly to the overall log-likelihood objective.
    //  Because their probabilities are multiplied in the model, their log-likelihoods are summed.  Here each individual pixel and the prior
    //  will have their individual values returned
    // NumComponets is prod(ImageSize) for MLE models and prod(ImageSize)+NumParams for MAP models where each of the final components corresponds to a single parameter in the
    //
    // (in) image: an image, double size:[imsizeY,imsizeX]
    // (in) theta: a parameter value size:[NumParams,1] double of theta
    // (out) rllh: relative log likelihood components size:[1,NumComponents]
    // (out) grad: (optional) components of grad of log likelihood size:[NumParams,NumComponents*]  * there is only a single extra grad component added for prior in MAP models.
    // (out) hess: (optional) hessian of log likelihood double size:[NumParams,NumParams,NumComponents*] * there is only a single extra hess component added for prior in MAP models.
    checkMinNumArgs(1,2);
    checkMaxNumArgs(3,2);
    auto im = getNumeric<ImageShapeT,ImagePixelT>();
    auto theta = getVec();
    output(methods::objective::rllh_components(*obj,im,theta));
    if(nlhs>=2) output(methods::objective::grad_components(*obj,im,theta));
    if(nlhs==3) output(methods::objective::hessian_components(*obj,im,theta));
}


//Static debugging for cholesky

template<class Model>
void Mappel_IFace<Model>::staticCholesky()
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
void Mappel_IFace<Model>::staticModifiedCholesky()
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
void Mappel_IFace<Model>::staticCholeskySolve()
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

template<class Model>
void Mappel_IFace<Model>::staticPositiveDefiniteCholeskyApprox()
{
    checkNumArgs(1,1);
    auto m = getMat();
    auto m_definite = makeOutputArray(m.n_rows, m.n_cols);
    m_definite = m;
    cholesky_make_positive_definite(m_definite);
}

template<class Model>
void Mappel_IFace<Model>::staticNegativeDefiniteCholeskyApprox()
{
    checkNumArgs(1,1);
    auto m = getMat();
    auto m_definite = makeOutputArray(m.n_rows, m.n_cols);
    m_definite = m;
    cholesky_make_negative_definite(m_definite);
}

template<class Model>
void MappelFixedSigma_IFace<Model>::objSetPSFSigma()
{
    this->checkNumArgs(0,1);
    obj->set_psf_sigma(this->getVec());
}

template<class Model>
void MappelVarSigma_IFace<Model>::objSetMinSigma()
{
    this->checkNumArgs(0,1);
    obj->set_min_sigma(this->getVec());
}

template<class Model>
void MappelVarSigma_IFace<Model>::objSetMaxSigma()
{
    this->checkNumArgs(0,1);
    obj->set_max_sigma(this->getVec());
}


// } /* namespace mappel */

#endif /* MAPPEL_MAPPEL_IFACE_H */
