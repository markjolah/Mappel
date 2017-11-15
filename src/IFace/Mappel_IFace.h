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

using namespace mexiface;
namespace mappel {

template<class Model>
class Mappel_IFace :  public MexIFace, public MexIFaceHandler<Model> {
public:
    using Stencil = typename Model::Stencil;
    using ParamT = typename Model::ParamT;
    using ParamVecT = typename Model::ParamVecT;
    using ParamMatT = typename Model::ParamMatT;
    using ImageT = typename Model::ImageT;
    using ImageStackT = typename Model::ImageStackT;

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
    void objCRLB();
    void objFisherInformation();
    void objModelGrad();
    void objModelHessian();
    void objModelObjective();
    void objModelPositiveHessian();
    void objEstimate();
    void objEstimateDebug();
    void objEstimatePosterior();
    void objEstimatePosteriorDebug();
        
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
    methodmap["samplePrior"]=boost::bind(&Mappel_IFace::objSamplePrior, this);
    methodmap["modelImage"]=boost::bind(&Mappel_IFace::objModelImage, this);
    methodmap["simulateImage"]=boost::bind(&Mappel_IFace::objSimulateImage, this);
    methodmap["LLH"]=boost::bind(&Mappel_IFace::objLLH, this);
    methodmap["modelGrad"]=boost::bind(&Mappel_IFace::objModelGrad, this);
    methodmap["modelHessian"]=boost::bind(&Mappel_IFace::objModelHessian, this);
    methodmap["modelObjective"]=boost::bind(&Mappel_IFace::objModelObjective, this);
    methodmap["modelPositiveHessian"]=boost::bind(&Mappel_IFace::objModelPositiveHessian, this);
    methodmap["CRLB"]=boost::bind(&Mappel_IFace::objCRLB, this);
    methodmap["fisherInformation"]=boost::bind(&Mappel_IFace::objFisherInformation, this);
    methodmap["estimate"]=boost::bind(&Mappel_IFace::objEstimate, this);
    methodmap["estimateDebug"]=boost::bind(&Mappel_IFace::objEstimateDebug, this);
    methodmap["estimatePosterior"]=boost::bind(&Mappel_IFace::objEstimatePosterior, this);
    methodmap["estimatePosteriorDebug"]=boost::bind(&Mappel_IFace::objEstimatePosteriorDebug, this);
    methodmap["getStats"]=boost::bind(&Mappel_IFace::objGetStats, this);
    methodmap["thetaInBounds"]=boost::bind(&Mappel_IFace::objThetaInBounds, this);
    methodmap["boundTheta"]=boost::bind(&Mappel_IFace::objBoundTheta, this);
    
    staticmethodmap["cholesky"]=boost::bind(&Mappel_IFace::objCholesky, this);
    staticmethodmap["modifiedCholesky"]=boost::bind(&Mappel_IFace::objModifiedCholesky, this);
    staticmethodmap["choleskySolve"]=boost::bind(&Mappel_IFace::objCholeskySolve, this);
}

template<class Model>
void Mappel_IFace<Model>::objGetHyperparams()
{
    // obj.SetHyperparams(prior)
    //(in) prior: 1x4 double - [Imin, Imax, bgmin, bgmax]
    checkNumArgs(1,0);
    output(obj->get_hyperparams());
}

template<class Model>
void Mappel_IFace<Model>::objSetHyperparams()
{
    // obj.SetHyperparams(prior)
    //(in) prior: 1x4 double - [Imin, Imax, bgmin, bgmax]
    checkNumArgs(0,1);
    obj->set_hyperparams(getVec());
}

template<class Model>
void Mappel_IFace<Model>::objSamplePrior()
{
    // theta=obj.samplePrior(count)
    // % (in) count: uint64 (default 1) number of thetas to sample
    // % (out) theta: A (nParams X n) double of theta values
    checkNumArgs(1,1);
    auto count = getAsInt<IdxT>();
    auto theta = makeOutputArray(obj->get_num_params(), count);
    sample_prior_stack(*obj, theta);
}

template<class Model>
void Mappel_IFace<Model>::objModelImage()
{
    // image=obj.modelImage(theta)
    // (in) theta: an (nParams X n) double of theta values
    // (out) image: a (size X size X n) double image stack
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto image_stack = obj->make_image_stack(theta_stack.n_cols);
    model_image_stack(*obj, theta_stack, image_stack);
}

template<class Model>
void Mappel_IFace<Model>::objSimulateImage()
{
    // image=obj.simulateImage(theta, count)
    // If theta is size (nParams X 1) then count images with that theta are
    // simulated.  Default count is 1.  If theta is size (nParams X n) with n>1
    // then n images are simulated, each with a seperate theta, and count is ignored.
    // (in) theta: a single (nParams X 1) or (nParams X n) double theta value.
    // (in) count: the number of independant images to generate
    // (out) image: a double (size X size X n) image stack, all sampled with params theta
    checkNumArgs(1,2);
    auto theta_stack = getMat();
    IdxT count = theta_stack.n_cols;
    if (count==1) count = getAsInt<IdxT>();
    auto image_stack = obj->make_image_stack(count);
    simulate_image_stack(*obj, theta_stack, image_stack);
}

template<class Model>
void Mappel_IFace<Model>::objLLH()
{
    // llh=obj.LLH(image, theta)
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single LLH.  If there are N=1 images
    // and M>1 thetas, we return M LLHs of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N LLHs for each of the images given theta
    // (in) image: a double (imsize X imsize X N) image stack
    // (in) theta: an (nParams X M) double of theta values
    // (out) llh: a (1 X max(M,N)) double of log_likelihoods
    checkNumArgs(1,2);
    auto image_stack = MexIFace::get<ImageStackT,double>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto llh_stack = makeOutputArray(count);
    log_likelihood_stack(*obj, image_stack, theta_stack, llh_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelGrad()
{
    // grad=obj.modelGrad(obj, image, theta) - Compute the model gradiant.
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single Grad.  If there are N=1 images
    // and M>1 thetas, we return M Grads of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N Grads for each of the images given theta
    // (in) image: a double (imsize X imsize X N) image stack
    // (in) theta: an (nParams X M) double of theta values
    // (out) grad: a (nParams X max(M,N)) double of gradiant vectors
    checkNumArgs(1,2);
    auto image_stack = get<ImageStackT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto grad_stack = makeOutputArray(obj->get_num_params(), count);
    model_grad_stack(*obj, image_stack, theta_stack, grad_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelHessian()
{
    // hess=obj.modelHessian(obj, image, theta) - Compute the model hessian
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single Hessian.  If there are N=1 images
    // and M>1 thetas, we return M Hessian of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N Hessians for each of the images given theta
    // (in) image: a double (imsize X imsize X N) image stack
    // (in) theta: an (nParams X M) double of theta values
    // (out) hess: a (nParams X nParams X max(M,N)) double of hessian matricies
    checkNumArgs(1,2);
    auto image_stack = get<ImageStackT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto hess_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(), count);
    model_hessian_stack(*obj, image_stack, theta_stack, hess_stack);
}

template<class Model>
void Mappel_IFace<Model>::objModelObjective()
{
    // hess=obj.modelObjective(obj, image, theta) - 
    // A convenience function for objective based optimization.  Works on a single image, theta and shares the
    // stencil to compute the LLH,Grad,Hessian as the 3 outputs.
    // This allows faster use with matlab optimization.
    // [in] image: an image, double size:[imsizeY,imsizeX] 
    // [in] theta: a parameter value size:[nParams,1] double of theta
    // [in] (optional) negate: boolean. true if objective should be negated, as is the case with
    //                matlab minimization routines
    // [out] LLH:  log likelihood scalar double 
    // [out] (optional) Grad: grad of log likelihood scalar double size:[nParams,1] 
    // [out] (optional) Hess: hessian of log likelihood double size:[nParams,nParams] 
    checkMinNumArgs(1,3);
    checkMaxNumArgs(3,3);
    auto image = get<ImageT>();
    auto theta = getVec();
    bool negate = getAsBool();
    auto stencil = obj->make_stencil(theta);
    double llh = log_likelihood(*obj, image, stencil);
    if(negate) llh = -llh;
    output(llh);
    if(nlhs==2) {
        //nargout=2 Output grad also
        auto grad = makeOutputArray(obj->get_num_params());
        model_grad(*obj, image, stencil, grad);
        if(negate) grad = -grad;
    } else if(nlhs==3) {
        //nargout=3 Output both grad and hess which can be computed simultaneously!
        auto grad = makeOutputArray(obj->get_num_params());
        auto hess = makeOutputArray(obj->get_num_params(),obj->get_num_params());
        model_hessian(*obj, image, stencil, grad, hess);
        copy_Usym_mat(hess);
        if(negate) grad = -grad;
        if(negate) hess = -hess;
    }
}

template<class Model>
void Mappel_IFace<Model>::objModelPositiveHessian()
{
    // hess=obj.modelPositiveHessian(obj, image, theta) - Compute 
    //  the modified cholesky decomposition form of the negative hessian which
    // should be positive definite at the maximum.  Equivalently the modified 
    // hessian itself is negative definite.
    //
    // This takes in a N images and M thetas.  If M=N=1, 
    // then we return a single Hessian.  If there are N=1 images
    // and M>1 thetas, we return M Hessian of the same image with each of 
    // the thetas.  Otherwise, if there is M=1 thetas and N>1 images,
    // then we return N Hessians for each of the images given theta
    // (in) image: a double (imsize X imsize X N) image stack
    // (in) theta: an (nParams X M) double of theta values
    // (out) hess: a (nParams X nParams X max(M,N)) double of positive hessian matricies
    checkNumArgs(1,2);
    auto image_stack = get<ImageStackT>();
    auto theta_stack = getMat();
    auto count = std::max(theta_stack.n_cols, obj->size_image_stack(image_stack));
    auto hess_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(), count);
    model_positive_hessian_stack(*obj, image_stack, theta_stack, hess_stack);
}

template<class Model>
void Mappel_IFace<Model>::objCRLB()
{
    // crlb=obj.CRLB(obj, theta) - Compute the Cramer-Rao Lower Bound at theta
    // (in) theta: an (nParams X n) double of theta values
    // (out) crlb: a  (nParams X n) double of the cramer rao lower bounds.
    //             these are the lower bounds on the variance at theta 
    //             for any unbiased estimator.
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto crlb_stack = makeDMat(obj->get_num_params(),theta_stack.n_cols);
    cr_lower_bound_stack(*obj, theta_stack, crlb_stack);
}

template<class Model>
void Mappel_IFace<Model>::objFisherInformation()
{
    // fisherI=obj.FisherInformation(obj, theta) - Compute the Fisher Information matrix
    //    at theta
    // (in) theta: an (nParams X n) double of theta values
    // (out) fisherI: a  (nParams X nParms) matrix of the fisher information at theta 
    checkNumArgs(1,1);
    auto theta_stack = getMat();
    auto fisherI_stack = makeOutputArray(obj->get_num_params(),obj->get_num_params(),theta_stack.n_cols);
    fisher_information_stack(*obj, theta_stack, fisherI_stack);
}

template<class Model>
void Mappel_IFace<Model>::objEstimate()
{
    // [theta, crlb, llh]=obj.estimate(image, name) - estimate theta's
    // crlb's and llh's for each image in stack.  
    // (in) image: a double (imsize X imsize X n) image stack
    // (in) name: (optional) name for the optimization method. (default = 'Newton')  
    //      Valid names are in obj.EstimationMethods
    // (in) theta_init: (optional) a (nParams x n) double of initial theta values to use in optimization
    // (out) theta: a (nParams X n) double of estimated theta values
    // (out) crlb: a (nParams X n) estimate of the CRLB for each parameter estimate.
    //             This gives the approximiate variance in the theta parameters
    // (out) llh: a (1 X n) double of the log likelihood at each theta estimate.
    // (out) stats: A 1x1 struct of fitting statistics.
    checkMaxNumArgs(4,3);
    //Get input
    auto image_stack = get<ImageStackT>();
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
        throw BadInputError(out.str());
    }
    estimator->estimate_stack(image_stack, theta_init_stack, theta_stack, crlb_stack, llh_stack);
    output(estimator->get_stats());
}

template<class Model>
void Mappel_IFace<Model>::objEstimateDebug()
{
    //  [theta, crlb, llh, stats, sample, sample_rllh]=estimatedebug(image, name) - estimate theta's
    //  crlb's and llh's for each image in stack.  
    //  (in) image: a (imsize X imsize) image
    //  (in) estimator_name: (optional) name for the optimization method. (default = 'Newton')  
    //            Valid names are in obj.EstimationMethods
    //  (in) theta_init: (optional) a (nParams x 1) double of initial theta values to use in optimization
    //  (out) theta: a (nParams X 1) double of estimated theta values
    //  (out) crlb: a (nParams X 1) estimate of the CRLB for each parameter estimate.
    //              This gives the approximiate variance in the theta parameters
    //  (out) llh: a (1 X 1) double of the log likelihood at each theta estimate.
    //  (out) stats: A 1x1 struct of fitting statistics.
    //  (out) sample: A (nParams X n) array of thetas that were searched as part of the maximization process
    //  (out) sample_llh: A (1 X n) array of relative log likelyhoods at each sample theta
    checkNumArgs(6,3);
    //Get input
    auto image = get<ImageT>();
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
        throw BadInputError(out.str());
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
    auto ims = get<ImageStackT>();
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
    auto im = get<ImageT>();
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

template<class Model>
void Mappel_IFace<Model>::objGetStats()
{
    checkNumArgs(1,0);
    output(obj->get_stats());
}

template<class Model>
void Mappel_IFace<Model>::objThetaInBounds()
{
    checkNumArgs(1,1);
    auto param = getMat();
    bool ok = true;
    for(IdxT i=0; i<param.n_cols; i++) ok &= obj->theta_in_bounds(param.col(i));
    output(ok);
}

template<class Model>
void Mappel_IFace<Model>::objBoundTheta()
{
    checkNumArgs(1,1);
    auto param = getMat();
    auto bounded = makeOutputArray(param.n_rows, param.n_cols);
    auto theta = obj->make_param();
    for(IdxT i=0; i<param.n_cols; i++) {
        theta = param.col(i);
        obj->bound_theta(theta);
        bounded.col(i) = theta;
    }
}

//Static debugging for cholesky

template<class Model>
void Mappel_IFace<Model>::objCholesky()
{
    checkNumArgs(3,1);
    auto A = getMat();
    if(!is_symmetric(A)) error("InvalidInput","Matrix is not symmetric");
    auto C = A;
    bool valid = cholesky(C);
    //Seperate d from C so C is unit lower triangular
    auto d = C.diag();
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
    if(!is_symmetric(A)) error("InvalidInput","Matrix is not symmetric");
    auto C = A;
    bool modified = modified_cholesky(C);
    auto d = C.diag();
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
    if(!is_symmetric(A)) error("InvalidInput","Matrix is not symmetric");
    if(b.n_elem != A.n_rows) error("InvalidInput","Input sizes do not match");
    auto C = A;
    bool modified = modified_cholesky(C);
    auto x = cholesky_solve(C,b);
    output(modified);
    output(x);
}

} /* namespace mappel */

#endif /* _MAPPEL_IFACE */
