
/** @file model_methoods_impl.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief Methods definitions for the model:: namespace which contains the major methods for computing with PointEmitterModels
 */

#ifndef _MAPPEL_MODEL_METHODS_IMPL_H
#define _MAPPEL_MODEL_METHODS_IMPL_H

#include "numerical.h"

namespace mappel {

namespace methods {
    template<class Model>
    typename Model::ImageT model_image(const Model &model, const ParamT<Model> &theta) 
    {
        return model_image(model, model.make_stencil(theta,false)); //don't compute derivative stencils
    }

    template<class Model>
    ModelDataT<Model> simulate_image(const Model &model, const ParamT<Model> &theta) 
    {
        return simulate_image(model, model.make_stencil(theta,false), rng_manager.generator()); //don't compute derivative stencils
    }
    
    template<class Model, class rng_t>
    ModelDataT<Model> simulate_image(const Model &model, const ParamT<Model> &theta, rng_t &rng) 
    {
        return simulate_image(model, model.make_stencil(theta,false), rng); //don't compute derivative stencils
    }

    template<class Model>
    ModelDataT<Model> simulate_image(const Model &model, const StencilT<Model> &s)
    {
        return simulate_image(model,s,rng_manager.generator()); //Make new generator
    }

    template<class Model>
    ModelDataT<Model> simulate_image_from_model(const Model &model, const ImageT<Model> &model_im)
    {
        return simulate_image_from_model(model,model_im,rng_manager.generator()); //Make new generator
    }
    
    namespace objective {            
        template<class Model>
        double 
        llh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            return llh(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils 
        }
      
        template<class Model>
        double 
        rllh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            return rllh(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils
        }

        template<class Model>
        ParamT<Model> 
        grad(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        {
            return grad(model, data_im, model.make_stencil(theta));
        }

        template<class Model>
        ParamT<Model> 
        grad2(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        {
            auto grad_val = model.make_param(); //Ignore un-requested value
            auto grad2_val = model.make_param();
            grad2(model, data_im, model.make_stencil(theta), grad_val, grad2_val);
            return grad2_val;
        }

        template<class Model>
        void
        grad2(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
              ParamT<Model> &grad_val, ParamT<Model> &grad2_val)
        {
            grad2(model, data_im, model.make_stencil(theta), grad_val, grad2_val);
        }

        template<class Model>
        MatT 
        hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            return hessian(model,data_im, model.make_stencil(theta)); 
        }

        template<class Model>
        MatT 
        hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            auto grad = model.make_param(); //Ignore un-requested value
            auto hess = model.make_param_mat();
            hessian(model, data_im, s, grad, hess);
            copy_Usym_mat(hess); //Make a full-symmetric matrix
            return hess;
        }
        
        template<class Model>
        void 
        hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta,
                ParamT<Model> &grad, MatT &hess)
        {
            hessian(model, data_im, model.make_stencil(theta), grad, hess);
        }
        



        template<class Model>
        MatT 
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            return negative_definite_hessian(model, data_im, model.make_stencil(theta));
        }

        template<class Model>
        MatT 
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            auto grad = model.make_param(); //Ignore un-requested value
            auto hess = model.make_param_mat();
            negative_definite_hessian(model, data_im, s, grad, hess);
            copy_Usym_mat(hess); //Make a full-symmetric matrix
            return hess;
        }
        
        template<class Model>
        void
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta,
                                  ParamT<Model> &grad, MatT &hess)
        {
            negative_definite_hessian(model, data_im, model.make_stencil(theta), grad, hess);
        }
        
        template<class Model>
        void
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s,
                                  ParamT<Model> &grad, MatT &hess)
        {
            hessian(model, data_im, s, grad, hess);
            hess = -hess;
            modified_cholesky(hess);
            cholesky_convert_full_matrix(hess); //convert from internal format to a full (negative definite) matrix
            hess = -hess;
        }

        inline namespace debug {
            template<class Model>
            VecT 
            llh_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &theta)
            { 
                return llh_components(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils 
            }

            template<class Model>
            VecT 
            rllh_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &theta)
            { 
                return rllh_components(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils
            }

            template<class Model>
            MatT
            grad_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &theta)
            {
                return grad_components(model, data_im, model.make_stencil(theta));
            }
            
            template<class Model>
            MatT
            hessian_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &theta)
            {
                return hessian_components(model, data_im, model.make_stencil(theta));
            }
        } /* mappel::methods::objective::debug */
    } /* mappel::methods::objective */

    
    template<class Model>
    void 
    aposteriori_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        rllh = likelihood::rllh(model, data_im, s);
        likelihood::hessian(model, data_im, s, grad, hess);
        rllh += model.prior.rllh(s.theta);
        model.prior.grad_hess_accumulate(s.theta,grad,hess);
        copy_Usym_mat(hess);
    }

    template<class Model>
    void 
    prior_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                    double &rllh, ParamT<Model> &grad, MatT &hess)
    {
        grad.zeros();
        hess.zeros();
        rllh = model.prior.rllh(s.theta);
        model.prior.grad_hess_accumulate(s.theta,grad,hess);
        copy_Usym_mat(hess);
    }

    template<class Model>
    void 
    likelihood_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                         double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        rllh = likelihood::rllh(model, data_im, s);
        likelihood::hessian(model, data_im, s, grad, hess);
        copy_Usym_mat(hess); //Accumulate symmetric matrix in upper triangular form
    }

    template<class Model>
    void 
    aposteriori_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        aposteriori_objective(model,data_im,model.make_stencil(theta),rllh,grad,hess);
    }

    template<class Model>
    void 
    prior_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        prior_objective(model,data_im,model.make_stencil(theta),rllh,grad,hess);
    }
    
    template<class Model>
    void 
    likelihood_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        likelihood_objective(model,data_im,model.make_stencil(theta),rllh,grad,hess);
    }

    template<class Model>
    ParamT<Model> cr_lower_bound(const Model &model, const typename Model::Stencil &s)
    {
        auto FI = expected_information(model,s);
        try{
            return arma::pinv(FI).eval().diag();
        } catch ( std::runtime_error E) {
            std::cout<<"Got bad fisher_information!!\n"<<"theta:"<<s.theta.t()<<"\n FI: "<<FI<<'\n';
            auto z = model.make_param();
            z.zeros();
            return z;
        }
    }

    template<class Model>
    ParamT<Model> cr_lower_bound(const Model &model, const ParamT<Model> &theta) 
    {
        return cr_lower_bound(model,model.make_stencil(theta));
    }

    template<class Model>
    MatT expected_information(const Model &model, const ParamT<Model> &theta) 
    {
        return expected_information(model,model.make_stencil(theta));
    }

    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta_mode)
    {
        auto hess = objective::hessian(model,data,theta_mode);
        hess = - hess; //Observed information is defined for negative llh and so negative hessian should be positive definite
        if(!is_positive_definite(hess)) throw NumericalError("Hessian is not positive definite");
        copy_Usym_mat(hess);
    }

    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_mode)
    {
        return observed_information(model,data, model.make_stencil(theta_mode));
    }
} /* namespace mappel::methods */
    

} /* namespace mappel */

#endif /* _MAPPEL_MODEL_METHODS_IMPL_H */
