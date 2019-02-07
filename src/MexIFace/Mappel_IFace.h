/** @file Mappel_IFace.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for Mappel_IFace.
 */

#ifndef MAPPEL_MAPPEL_IFACE_H
#define MAPPEL_MAPPEL_IFACE_H

#include <sstream>
#include <iostream>
#include <functional>

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
    void objSetPSFSigma();
    void objGetBounds();
    void objSetBounds();

    void objBoundTheta();
    void objThetaInBounds();

    void objGetStats();
    void objSamplePrior();
    
    void objModelImage();
    void objSimulateImage();
    
    void objModelLLH();
    void objModelGrad();
    void objModelHessian();
    
    void objModelObjective();
    void objModelObjectiveAPosteriori();
    void objModelObjectiveLikelihood();
    void objModelObjectivePrior();

    void objExpectedInformation();
    void objCRLB();

    void objEstimate();
    //void objEstimateProfile();
    void objEstimatePosterior();

    void objErrorBoundsObserved();
    void objErrorBoundsExpected();
    void objErrorBoundsProfileLikelihood();
    void objErrorBoundsPosteriorCredible();


    /* Degugging */    
    void objEstimateDebug();
    void objEstimatePosteriorDebug();
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
    //These are used to set properties.
    methodmap["getHyperparams"] = std::bind(&Mappel_IFace::objGetHyperparams, this);
    methodmap["setHyperparams"] = std::bind(&Mappel_IFace::objSetHyperparams, this);
    methodmap["getHyperparamNames"] = std::bind(&Mappel_IFace::objGetHyperparamNames, this);
    methodmap["setHyperparamNames"] = std::bind(&Mappel_IFace::objSetHyperparamNames, this);
    methodmap["getParamNames"] = std::bind(&Mappel_IFace::objGetParamNames, this);
    methodmap["setParamNames"] = std::bind(&Mappel_IFace::objSetParamNames, this);
    methodmap["setImageSize"] = std::bind(&Mappel_IFace::objSetImageSize, this);
    methodmap["setPSFSigma"] = std::bind(&Mappel_IFace::objSetPSFSigma, this);
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
    methodmap["modelGrad"] = std::bind(&Mappel_IFace::objModelGrad, this);
    methodmap["modelHessian"] = std::bind(&Mappel_IFace::objModelHessian, this);

    methodmap["modelObjective"] = std::bind(&Mappel_IFace::objModelObjective, this);
    methodmap["modelObjectiveAPosteriori"] = std::bind(&Mappel_IFace::objModelObjectiveAPosteriori, this);
    methodmap["modelObjectiveLikelihood"] = std::bind(&Mappel_IFace::objModelObjectiveLikelihood, this);
    methodmap["modelObjectivePrior"] = std::bind(&Mappel_IFace::objModelObjectivePrior, this);

    methodmap["expectedInformation"] = std::bind(&Mappel_IFace::objExpectedInformation, this);
    methodmap["CRLB"] = std::bind(&Mappel_IFace::objCRLB, this);

    methodmap["estimate"] = std::bind(&Mappel_IFace::objEstimate, this);
    //methodmap["estimateProfile"] = std::bind(&Mappel_IFace::objEstimateProfile, this);
    methodmap["estimatePosterior"] = std::bind(&Mappel_IFace::objEstimatePosterior, this);

    methodmap["errorBoundsObserved"] = std::bind(&Mappel_IFace::objErrorBoundsObserved, this);
    methodmap["errorBoundsExpected"] = std::bind(&Mappel_IFace::objErrorBoundsExpected, this);
    methodmap["errorBoundsProfileLikelihood"] = std::bind(&Mappel_IFace::objErrorBoundsProfileLikelihood, this);

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
void Mappel_IFace<Model>::objSetPSFSigma()
{
    checkNumArgs(0,1);
    obj->set_psf_sigma(getVec());
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
    // (in) theta - double [NumParams, n] stack of thetas to bound
    // (out) in_bounds - bool.  True if all theta are in bounds.
    checkNumArgs(1,1);
    auto param = getMat();
    bool ok = true;
    for(IdxT i=0; i<param.n_cols; i++) ok &= obj->theta_in_bounds(param.col(i));
    output(ok);
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
    // (out) hess: double size:[NumParams,NumParams,max(M,N)] stack of hessian matricies
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
    // [rllh,grad,hess,llh] = obj.modelObjective(image, theta, negate) -
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
    auto stencil = obj->make_stencil(theta);
    double rllh = negate_scalar*methods::objective::rllh(*obj, image, stencil);
    if(negate) rllh = -rllh;
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
        grad *= negate_scalar;
        hess *= negate_scalar;
        if(nlhs>=4) {
            auto definite_hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
            definite_hess = hess;
            if(negate) {
                cholesky_make_positive_definite(definite_hess);
            } else {
                cholesky_make_negative_definite(definite_hess);
            }
        }
        if(nlhs==5) output(negate_scalar*methods::objective::llh(*obj, image, stencil)); //ouput optional full llh
    }
}


template<class Model>
void Mappel_IFace<Model>::objModelObjectiveAPosteriori()
{
    // [rllh,grad,hess,llh] = obj.modelObjectiveAPosteriori(image, theta, negate) -
    //
    // Evaluate the aposteriori objective irrepective of the model's MLE/MAP.
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
    double negate_scalar = negate ? -1 : 1;
    auto stencil = obj->make_stencil(theta);

    double rllh;
    auto grad = obj->make_param();
    auto hess = obj->make_param_mat();
    methods::aposteriori_objective(*obj, image, stencil, rllh, grad, hess);
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
        if(negate) {
            cholesky_make_positive_definite(definite_hess);
        } else {
            cholesky_make_negative_definite(definite_hess);
        }
    }
    if(nlhs>=5) output(methods::likelihood::llh(*obj, image, stencil) + obj->get_prior().llh(theta));
}

template<class Model>
void Mappel_IFace<Model>::objModelObjectiveLikelihood()
{
    // [rllh,grad,hess,llh] = obj.modelObjectiveLikelihood(image, theta, negate) -
    //
    // Evaluate the pure-likelihood based objective irrepective of the model's MLE/MAP.
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
    auto stencil = obj->make_stencil(theta);

    double rllh;
    auto grad = obj->make_param();
    auto hess = obj->make_param_mat();
    methods::likelihood_objective(*obj, image, stencil, rllh, grad, hess);
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
    // Evaluate the pure-prior likelihood based objective irrepective of the model's MLE/MAP.
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
        if(negate) {
            cholesky_make_positive_definite(definite_hess);
        } else {
            cholesky_make_negative_definite(definite_hess);
        }
    }
    if(nlhs>=5) output(obj->get_prior().llh(theta));
}

template<class Model>
void Mappel_IFace<Model>::objExpectedInformation()
{
    // fisherI = obj.expectedInformation(theta) - Compute the Expected (Fisher) Information matrix
    //    at theta
    // (in) theta: double size:[NumParams, n] stack of theta values
    // (out) fisherI: double size:[NumParams,nParms, n] stack if symmetric fisher information matricies at each theta
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
    // Retuns the observedInformation matrix which can be used to estimate the error 
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
    int nimages = obj->get_size_image_stack(image_stack);
    //Make output
    auto theta_stack = makeOutputArray(obj->get_num_params(),nimages);
    auto rllh_stack = makeOutputArray(nimages);
    auto obsI_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(),nimages);
    if(nlhs<5) {
        methods::estimate_max_stack(*obj, image_stack,method_name,theta_init_stack, theta_stack, rllh_stack, obsI_stack);
    } else {
        StatsT stats;
        methods::estimate_max_stack(*obj, image_stack,method_name,theta_init_stack, theta_stack, rllh_stack, obsI_stack,stats);
        output(stats);
    }
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
    checkNumArgs(6,3);
    //Get input
    auto image = getNumeric<ImageShapeT,ImagePixelT>();
    auto method_name = getString();
    auto theta_init = getVec();

    auto theta = obj->make_param();
    double rllh;
    auto obsI = obj->make_param_mat();
    MatT sequence;
    VecT sequence_rllh;
    StatsT stats;
    methods::estimate_max_debug(*obj, image, method_name, theta, rllh, obsI, sequence, sequence_rllh, stats);

    //Write output
    output(theta);
    output(rllh);
    output(obsI);
    output(sequence);
    output(sequence_rllh);
    output(stats); //Get the debug stats which might include more info like backtrack_idxs
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
    // (in) theta_init: (optional) Initial theta guesses size:[NumParams,1]. [default: [] ] Empty array to force auto estimation.
    //                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
    //                   should be auto estimated.
    // (in) confidence: (optional) desired confidence to estimate credible interval at.  Given as 0<confidence<1. [default=obj.DefaultConfidenceLevel]
    // (in) num_samples: (optional) Number of (post-filtering) posterior samples to aquire. [default=obj.DefaultMCMCNumSamples]
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
    checkMinNumArgs(6,6);
    //Get input
    auto ims = getNumeric<ImageStackShapeT,ImagePixelT>();
    auto theta_init_stack = getMat(); //Matlab iface code ensures we get proper shaped array even if user passes empty
    double confidence = getAsFloat();
    auto num_samples = getAsInt<IdxT>();
    auto burnin = getAsInt<IdxT>();
    auto thin = getAsInt<IdxT>();

    auto Nims = obj->get_size_image_stack(ims);
    auto Np = obj->get_num_params();//number of model parameters

    CubeT samples;
    MatT sample_rllh;

    methods::estimate_mcmc_sample_stack(*obj, ims, theta_init_stack, num_samples, burnin, thin, samples, sample_rllh);
    if(nlhs==1) {
        output(arma::mean(samples,1).eval()); //posterior mean is mean over each row
        return;
    }
    MatT theta_mean, theta_lb, theta_ub;
    methods::error_bounds_posterior_credible_stack(*obj, samples, confidence, theta_mean, theta_lb, theta_ub);
    if(nlhs>=2) output(theta_mean);
    if(nlhs>=3) output(theta_lb);
    if(nlhs>=4) {
        CubeT cov_stack(Np,Np,Nims);
        for(IdxT i=0; i<Nims; i++) {
            cov_stack.slice(i) = arma::cov(samples.slice(i).t());
        }
        output(cov_stack);
    }
    if(nlhs>=5) output(samples);
    if(nlhs==6) output(sample_rllh);
}

template<class Model>
void Mappel_IFace<Model>::objEstimatePosteriorDebug()
{
    // [sample, sample_llh, candidates, candidate_llh] = obj.estimatePosteriorDebug(image, theta_init, num_samples)
    //
    // Debugging routine.  Works on a single image.  Get out the exact MCMC sample sequence, as well as the candidate sequence.
    // Does not do burnin or thinning.
    //
    // (in) image: a sinle images to estimate
    // (in) theta_init: (optional) Initial theta guesses size:[NumParams,1]. [default: [] ] Empty array to force auto estimation.
    //                   Values of 0 for any individual parameter indicate that we have no initial guess for that parameter and it
    //                   should be auto estimated.
    // (in) num_samples: (optional) Number of (post-filtering) posterior samples to aquire. [default=obj.DefaultMCMCNumSamples]
    // (out) sample: A size:[NumParams,num_samples] array of thetas samples
    // (out) sample_llh: A size:[1,num_samples] array of log likelihoods at each sample theta
    // (out) candidates: (optional) size:[NumParams, num_samples] array of candidate thetas
    // (out) candidate_llh: (optional) A size:[1, num_samples] array of log likelihoods at each candidate theta
    checkNumArgs(4,3);
    //Get input
    auto im = getNumeric<ImageShapeT,ImagePixelT>(); //single image
    auto theta_init = getVec();
    auto Ns = getAsInt<IdxT>();
    auto Np = obj->get_num_params();//number of model parameters
    //Make ouput
    auto sample = makeOutputArray(Np,Ns);
    auto sample_llh = makeOutputArray(Ns);
    auto candidates = makeOutputArray(Np,Ns);
    auto candidate_llh = makeOutputArray(Ns);
    //Call method
    methods::estimate_mcmc_sample_debug(*obj,im,theta_init,Ns,sample,sample_llh,candidates,candidate_llh);
}


template<class Model>
void Mappel_IFace<Model>::objErrorBoundsObserved()
{
    // [observed_lb, observed_ub] = obj.errorBoundsObserved(image, theta_mle, confidence)
    //
    // Compute the error bounds using the observed information at the MLE estimate theta_mle.
    //
    // (in) image: a single image or a stack of n images
    // (in) theta_mle: the MLE estimated theta size:[NumParams,n]
    // (in) confidence: (optional) desired confidence as 0<p<1.  [default=obj.DefaultConfidenceLevel]
    // (in) obsI: (optional) observed information, at each theta_mle, as returned by obj.estimate size:[NumParams,NumParams,n]
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
        obsI*=-1;
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
    if(nlhs>2) output(methods::objective::grad_components(*obj,im,theta));
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

// } /* namespace mappel */

#endif /* MAPPEL_MAPPEL_IFACE_H */
