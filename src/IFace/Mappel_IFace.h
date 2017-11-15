/** @file Mappel_Iface.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for Mappel_Iface.
 */

#ifndef _MAPPEL_IFACE
#define _MAPPEL_IFACE
#include <sstream>
#include <iostream>
#include <typeinfo>
#include <type_traits>

#include "MexIFace.h"
#include "Handle.h"
#include "PointEmitterModel.h"

using namespace mexiface;
namespace mappel {

using std::cout;
using std::endl;

template<class Model>
class Mappel_Iface : public MexIFace {
public:
    typedef typename Model::Stencil Stencil;
    typedef typename Model::ParamT ParamT;
    typedef typename Model::ParamVecT ParamVecT;
    typedef typename Model::ParamMatT ParamMatT;
    typedef typename Model::ImageT ImageT;
    typedef typename Model::ImageStackT ImageStackT;
    Mappel_Iface(std::string name);

protected:
    Model *obj;

    virtual ImageT getImage()=0;
    virtual ImageStackT getImageStack()=0;
    virtual ImageStackT makeImageStack(int count)=0;

    void objDestroy();
    void objGetHyperparameters();
    void objSetHyperparameters();
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
    
    void objGetStats();
    void objThetaInBounds();
    void objBoundTheta();
    void getObjectFromHandle(const mxArray *mxhandle);
    
    void objCholesky();
    void objModifiedCholesky();
    void objCholeskySolve();
};

template<class Model>
Mappel_Iface<Model>::Mappel_Iface(std::string name) 
    : MexIFace(name)
{
    methodmap["getHyperparameters"]=boost::bind(&Mappel_Iface::objGetHyperparameters, this);
    methodmap["setHyperparameters"]=boost::bind(&Mappel_Iface::objSetHyperparameters, this);
    methodmap["samplePrior"]=boost::bind(&Mappel_Iface::objSamplePrior, this);
    methodmap["modelImage"]=boost::bind(&Mappel_Iface::objModelImage, this);
    methodmap["simulateImage"]=boost::bind(&Mappel_Iface::objSimulateImage, this);
    methodmap["LLH"]=boost::bind(&Mappel_Iface::objLLH, this);
    methodmap["modelGrad"]=boost::bind(&Mappel_Iface::objModelGrad, this);
    methodmap["modelHessian"]=boost::bind(&Mappel_Iface::objModelHessian, this);
    methodmap["modelObjective"]=boost::bind(&Mappel_Iface::objModelObjective, this);
    methodmap["modelPositiveHessian"]=boost::bind(&Mappel_Iface::objModelPositiveHessian, this);
    methodmap["CRLB"]=boost::bind(&Mappel_Iface::objCRLB, this);
    methodmap["fisherInformation"]=boost::bind(&Mappel_Iface::objFisherInformation, this);
    methodmap["estimate"]=boost::bind(&Mappel_Iface::objEstimate, this);
    methodmap["estimateDebug"]=boost::bind(&Mappel_Iface::objEstimateDebug, this);
    methodmap["estimatePosterior"]=boost::bind(&Mappel_Iface::objEstimatePosterior, this);
    methodmap["estimatePosteriorDebug"]=boost::bind(&Mappel_Iface::objEstimatePosteriorDebug, this);
    methodmap["getStats"]=boost::bind(&Mappel_Iface::objGetStats, this);
    methodmap["thetaInBounds"]=boost::bind(&Mappel_Iface::objThetaInBounds, this);
    methodmap["boundTheta"]=boost::bind(&Mappel_Iface::objBoundTheta, this);
    
    staticmethodmap["cholesky"]=boost::bind(&Mappel_Iface::objCholesky, this);
    staticmethodmap["modifiedCholesky"]=boost::bind(&Mappel_Iface::objModifiedCholesky, this);
    staticmethodmap["choleskySolve"]=boost::bind(&Mappel_Iface::objCholeskySolve, this);
}

template<class Model>
inline
void Mappel_Iface<Model>::getObjectFromHandle(const mxArray *mxhandle)
{
    obj=Handle<Model>::getObject(mxhandle);
}


template<class Model>
void Mappel_Iface<Model>::objDestroy()
{
    if(!nrhs) MappelError("BadInputArgs","Destructor: Bad object handle given");
    Handle<Model>::destroyObject(rhs[0]);
}


template<class Model>
void Mappel_Iface<Model>::objGetHyperparameters()
{
    // obj.SetHyperparameters(prior)
    //(in) prior: 1x4 double - [Imin, Imax, bgmin, bgmax]
    checkNumArgs(1,0);
    outputDVec(obj->get_hyperparameters());
}

template<class Model>
void Mappel_Iface<Model>::objSetHyperparameters()
{
    // obj.SetHyperparameters(prior)
    //(in) prior: 1x4 double - [Imin, Imax, bgmin, bgmax]
    checkNumArgs(0,1);
    obj->set_hyperparameters(getVec());
}

template<class Model>
void Mappel_Iface<Model>::objSamplePrior()
{
    // theta=obj.samplePrior(count)
    // % (in) count: int (default 1) number of thetas to sample
    // % (out) theta: A (nParams X n) double of theta values
    checkNumArgs(1,1);
    int  count = getInt();
    auto theta = makeOutputArray(obj->num_params, count);
    sample_prior_stack(*obj, theta);
}

template<class Model>
void Mappel_Iface<Model>::objModelImage()
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
void Mappel_Iface<Model>::objSimulateImage()
{
    // image=obj.simulateImage(theta, count)
    // If theta is size (nParams X 1) then count images with that theta are
    // simulated.  Default count is 1.  If theta is size (nParams X n) with n>1
    // then n images are simulated, each with a seperate theta, and count is ignored.
    // (in) theta: a single (nParams X 1) or (nParams X n) double theta value.
    // (in) count: the number of independant images to generate
    // (out) image: a double (size X size X n) image stack, all sampled with params theta
    checkNumArgs(1,2);
    auto theta_stack=getDMat();
    int  count=theta_stack.n_cols;
    if (count==1) count=getInt();
    auto image_stack=makeImageStack(count);
    simulate_image_stack(*obj, theta_stack, image_stack);
}

template<class Model>
void Mappel_Iface<Model>::objLLH()
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
    auto image_stack=getImageStack();
    auto theta_stack=getDMat();
    int  count=std::max(static_cast<int>(theta_stack.n_cols), static_cast<int>(image_stack.n_slices));
    auto llh_stack=makeDVec(count);
    log_likelihood_stack(*obj, image_stack, theta_stack, llh_stack);
}

template<class Model>
void Mappel_Iface<Model>::objModelGrad()
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
    auto image_stack=getImageStack();
    auto theta_stack=getDMat();
    int  count=std::max(static_cast<int>(theta_stack.n_cols), static_cast<int>(image_stack.n_slices));
    auto grad_stack=makeDMat(obj->num_params, count);
    model_grad_stack(*obj, image_stack, theta_stack, grad_stack);
}

template<class Model>
void Mappel_Iface<Model>::objModelHessian()
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
    auto image_stack=getImageStack();
    auto theta_stack=getDMat();
    int  count=std::max(static_cast<int>(theta_stack.n_cols), static_cast<int>(image_stack.n_slices));
    auto hess_stack=makeDStack(obj->num_params,obj->num_params, count);
    model_hessian_stack(*obj, image_stack, theta_stack, hess_stack);
}


template<class Model>
void Mappel_Iface<Model>::objModelObjective()
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
    auto image = getImage();
    auto theta = getDVec();
    bool negate = getBool();
    auto stencil = obj->make_stencil(theta);
    double llh = log_likelihood(*obj, image, stencil);
    if(negate) llh = -llh;
    outputDouble(llh);
    if(nlhs==2) {
        //nargout=2 Output grad also
        auto grad=makeDVec(obj->num_params);
        model_grad(*obj, image, stencil, grad);
        if(negate) grad = -grad;
    } else if(nlhs==3) {
        //nargout=3 Output both grad and hess which can be computed simultaneously!
        auto grad=makeDVec(obj->num_params);
        auto hess=makeDMat(obj->num_params,obj->num_params);
        model_hessian(*obj, image, stencil, grad, hess);
        copy_Usym_mat(hess);
        if(negate) grad = -grad;
        if(negate) hess = -hess;
    }
}


template<class Model>
void Mappel_Iface<Model>::objModelPositiveHessian()
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
    auto image_stack=getImageStack();
    auto theta_stack=getDMat();
    int  count=std::max(theta_stack.n_cols, image_stack.n_slices);
    auto hess_stack=makeDStack(obj->num_params,obj->num_params, count);
    model_positive_hessian_stack(*obj, image_stack, theta_stack, hess_stack);
}



template<class Model>
void Mappel_Iface<Model>::objCRLB()
{
    // crlb=obj.CRLB(obj, theta) - Compute the Cramer-Rao Lower Bound at theta
    // (in) theta: an (nParams X n) double of theta values
    // (out) crlb: a  (nParams X n) double of the cramer rao lower bounds.
    //             these are the lower bounds on the variance at theta 
    //             for any unbiased estimator.
    checkNumArgs(1,1);
    auto theta_stack=getDMat();
    auto crlb_stack=makeDMat(obj->num_params,theta_stack.n_cols);
    cr_lower_bound_stack(*obj, theta_stack, crlb_stack);
}

template<class Model>
void Mappel_Iface<Model>::objFisherInformation()
{
    // fisherI=obj.FisherInformation(obj, theta) - Compute the Fisher Information matrix
    //    at theta
    // (in) theta: an (nParams X n) double of theta values
    // (out) fisherI: a  (nParams X nParms) matrix of the fisher information at theta 
    checkNumArgs(1,1);
    auto theta_stack=getDMat();
    auto fisherI_stack=makeDStack(obj->num_params,obj->num_params,theta_stack.n_cols);
    fisher_information_stack(*obj, theta_stack, fisherI_stack);
}



template<class Model>
void Mappel_Iface<Model>::objEstimate()
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
    auto image_stack=getImageStack();
    std::string name=getString();
    auto theta_init_stack = getDMat();
    ParamVecT theta_init;
    int nimages=image_stack.n_slices;
    //Make output
    auto theta_stack=makeDMat(obj->num_params,nimages);
    auto crlb_stack=makeDMat(obj->num_params,nimages);
    auto llh_stack=makeDVec(nimages);
    //Call method
    auto estimator=make_estimator(*obj, name);
    if(!estimator) {
        std::ostringstream out;
        out<<"Bad estimator name: "<<name;
        component_error("estimate","InputError",out.str());
    }
    estimator->estimate_stack(image_stack, theta_init_stack, theta_stack, crlb_stack, llh_stack);
    outputStatsToStruct(estimator->get_stats());
}

template<class Model>
void Mappel_Iface<Model>::objEstimateDebug()
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
    auto image=getImage();
    std::string name=getString();
    auto theta_init = getDVec();
    //Make temporaries (don't know sequence lenght ahead of call)
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
    auto estimator=make_estimator(*obj, name);
    if(!estimator) {
        std::ostringstream out;
        out<<"Bad estimator name: "<<name;
        component_error("estimateDebug","InputError",out.str());
    }
    estimator->estimate_debug(image, theta_init_p, theta, crlb, llh, sequence, sequence_llh);
    //Write output
    outputDVec(theta);
    outputDVec(crlb);
    outputDouble(llh);
    outputStatsToStruct(estimator->get_debug_stats()); //Get the debug stats which might include more info like backtrack_idxs
    outputDMat(sequence);
    outputDVec(sequence_llh);
}


template<class Model>
void Mappel_Iface<Model>::objEstimatePosterior()
{
    // [mean, cov, stats]=estimatePosterior(obj, image, max_iterations)
    // (in) image: a double (imsize X imsize X n) image stack
    // (in) Nsamples: A  number of samples to take
    // (in) theta_init: (optional) a (nParams x n) double of initial theta values to use for starting MCMC
    // (out) mean: a (nParams X n) double of estimated posterior mean values
    // (out) cov: a (nParams X nParams X n) estimate of the posterior covarience.
    checkNumArgs(2,3);
    //Get input
    auto ims=getImageStack();
    int Nsamples=getInt();
    auto theta_init_stack = getDMat();

    int Nims=ims.n_slices;
    int Np=obj->num_params;//number of model parameters
    //Make output
    auto means=makeDMat(Np,Nims);
    auto covs=makeDStack(Np,Np,Nims);
    //Call method
    evaluate_posterior_stack(*obj, ims, theta_init_stack, Nsamples, means, covs);
}

template<class Model>
void Mappel_Iface<Model>::objEstimatePosteriorDebug()
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
    auto im=getImage();
    int Ns=getInt(); //number of samples
    auto theta_init = getDVec();
    int Np=obj->num_params;//number of model parameters
    //Make ouput
    auto mean=makeDVec(Np);
    auto cov=makeDMat(Np,Np);
    auto sample=makeDMat(Np,Ns);
    auto sample_llh=makeDVec(Ns);
    auto candidates=makeDMat(Np,Ns);
    auto candidate_llh=makeDVec(Ns);
    //Call method
    evaluate_posterior_debug(*obj,im,theta_init,Ns,mean,cov,sample,sample_llh,candidates,candidate_llh);
}


template<class Model>
void Mappel_Iface<Model>::objGetStats()
{
    checkNumArgs(1,0);
    outputStatsToStruct(obj->get_stats());
}

template<class Model>
void Mappel_Iface<Model>::objThetaInBounds()
{
    checkNumArgs(1,1);
    auto param=getDMat();
    bool ok = true;
    for(unsigned i=0; i<param.n_cols; i++) ok &= obj->theta_in_bounds(param.col(i));
    outputBool(ok);
}

template<class Model>
void Mappel_Iface<Model>::objBoundTheta()
{
    checkNumArgs(1,1);
    auto param=getDMat();
    auto bounded=makeDMat(param.n_rows, param.n_cols);
    auto theta=obj->make_param();
    for(unsigned i=0; i<param.n_cols; i++) {
        theta=param.col(i);
        obj->bound_theta(theta);
        bounded.col(i)=theta;
    }
}

//Static debugging for cholesky

template<class Model>
void Mappel_Iface<Model>::objCholesky()
{
    checkNumArgs(3,1);
    auto A=getDMat();
    if(!is_symmetric(A)) error("InvalidInput","Matrix is not symmetric");
    arma::mat C = A;
    bool valid = cholesky(C);
    //Seperate d from C so C is unit lower triangular
    arma::vec d = C.diag();
    C.diag().fill(1.);
    outputBool(valid);
    outputDMat(C);
    outputDVec(d);
}

template<class Model>
void Mappel_Iface<Model>::objModifiedCholesky()
{
    checkNumArgs(3,1);
    auto A=getDMat();
    if(!is_symmetric(A)) error("InvalidInput","Matrix is not symmetric");
    arma::mat C = A;
    bool modified = modified_cholesky(C);
    arma::vec d = C.diag();
    C.diag().fill(1.);
    outputBool(modified);
    outputDMat(C);
    outputDVec(d);
}

template<class Model>
void Mappel_Iface<Model>::objCholeskySolve()
{
    checkNumArgs(2,2);
    auto A=getDMat();
    auto b=getDVec();
    if(!is_symmetric(A)) error("InvalidInput","Matrix is not symmetric");
    if(b.n_elem != A.n_rows) error("InvalidInput","Input sizes do not match");
    arma::mat C = A;
    bool modified = modified_cholesky(C);
    arma::vec x= cholesky_solve(C,b);
    outputBool(modified);
    outputDVec(x);
}

} /* namespace mappel */

#endif /* _MAPPEL_IFACE */
