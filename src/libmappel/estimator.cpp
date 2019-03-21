/** @file estimator.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief Non-templated estimator helper routines and static constants
 */

#include <cmath>
#include <iomanip>
#include <armadillo>
#include "Mappel/util.h"
#include "Mappel/numerical.h"
#include "Mappel/estimator.h"
#include "Mappel/estimator_helpers.h"

namespace mappel {
namespace estimator {

void ProfileBoundsData::initialize_arrays(IdxT Nparams)
{
    Nparams_est = estimated_idxs.n_elem;
    profile_lb.set_size(Nparams_est);
    profile_ub.set_size(Nparams_est);
    profile_points_lb.set_size(Nparams, Nparams_est);
    profile_points_ub.set_size(Nparams, Nparams_est);
    profile_points_lb_rllh.set_size(Nparams_est);
    profile_points_ub_rllh.set_size(Nparams_est);
}

void ProfileBoundsDataStack::initialize_arrays(IdxT Nparams)
{
    Nparams_est = estimated_idxs.n_elem;
    profile_lb.set_size(Nparams_est,Ndata);
    profile_ub.set_size(Nparams_est,Ndata);
    profile_points_lb.set_size(Nparams, Nparams_est,Ndata);
    profile_points_ub.set_size(Nparams, Nparams_est,Ndata);
    profile_points_lb_rllh.set_size(Nparams_est,Ndata);
    profile_points_ub_rllh.set_size(Nparams_est,Ndata);
}


/** Estimation subroutines common to several estimators and independent of the Model
 */
namespace subroutine {

static const double boundary_stepback_min_kappa = 1.0 - 1.0e-2; ///<Distance to step back from the boundary to remain in interior
static const double trust_radius_init_min = 1.0e-3;///< Minimum initial trust region radius
static const double trust_radius_init_max = 1.0e3; ///< Maximum initial trust region radius
static const double min_scaling = 1.0e-5;///<Minimum for problem scaling via Dscale(i)
static const double max_scaling = 1.0e5;///<Maximum for problem scaling via Dscale(i)
static const double tr_subproblem_newton_epsilon = 1.0e-4;  ///< Convergence epsilon for newton's method usage in solve_restricted_step_length_newton()

VecT solve_profile_initial_step(const MatT &obsI, IdxT fixed_idx, double llh_delta)
{
    auto Np = obsI.n_rows;
    MatT free_hess(Np-1,Np-1);
    VecT fixed_grad2(Np-1);
    //obsI and free_hess are in symmetric upper-triangular format
    for(IdxT j=0,jj=0; j<Np; j++) {
        if(j!=fixed_idx) {
            for(IdxT i=0,ii=0;i<=j;i++) {
                if(i!=fixed_idx)
                    free_hess(ii++,jj) = -obsI(i,j);
            }
            jj++;
        } else {
            for(IdxT i=0,ii=0;i<Np;i++) {
                if(i!=fixed_idx)
                    fixed_grad2(ii++) = (i<j) ? -obsI(i,j) : -obsI(j,i);
            }
        }
    }

    VecT tangent = arma::solve(arma::symmatu(free_hess),-fixed_grad2);
    double denom = obsI(fixed_idx,fixed_idx)+arma::dot(fixed_grad2,tangent);
    double h = -sqrt(fabs(llh_delta/denom));

    VecT step(Np);
    for(IdxT i=0,ii=0; i<Np; i++) step(i) = (i==fixed_idx) ? h : h*tangent(ii++);
    return step;
}

VecT bound_step(const VecT &step, const VecT &theta, const VecT &lbound, const VecT &ubound)
{
    VecT step_ratio = theta/step;
    double alpha = arma::min( VecT(arma::max(lbound/step, ubound/step)-step_ratio) );
    double kappa_min = 0.9;
    if(alpha>1){ //step is feasible.  accept it
        VecT full_step = theta + step;
        if(arma::all(full_step > lbound) && arma::all(full_step < ubound)) return step;
    }
    //backtrack a little bit from alpha to remain feasible
    double kappa = boundary_stepback_min_kappa;
    VecT full_step = theta + kappa*std::min(alpha,1.)*step;
    while(!arma::all(full_step > lbound) || !arma::all(full_step < ubound)){
        kappa*=boundary_stepback_min_kappa;
        full_step = theta + kappa*std::min(alpha,1.)*step;
        std::cout<<"Kappa stepped back to: "<<kappa<<std::endl;
        if(kappa<kappa_min){
            std::cout<<std::setprecision(16);
            std::cout<<"alpha:"<<alpha<<" kappa:"<<kappa<<std::endl;
            step.t().raw_print(std::cout,"step:");
            theta.t().raw_print(std::cout,"theta:");
            full_step.t().raw_print(std::cout,"fullstep:");
            lbound.t().raw_print(std::cout,"lbound:");
            ubound.t().raw_print(std::cout,"ubound:");
            throw NumericalError("Kappa backing up did not correct bad step.");
        }
    }
    if(!arma::all(full_step > lbound)) {
        std::cout<<std::setprecision(16);
        std::cout<<"alpha:"<<alpha<<" kappa:"<<kappa<<std::endl;
        step.t().raw_print(std::cout,"step:");
        theta.t().raw_print(std::cout,"theta:");
        full_step.t().raw_print(std::cout,"fullstep:");
        lbound.t().raw_print(std::cout,"lbound:");
        ubound.t().raw_print(std::cout,"ubound:");
        throw NumericalError("bound_step: Infeasible Bounding failed lower bounds");
    }
    if(!arma::all(full_step < ubound)) {
        std::cout<<std::setprecision(16);
        std::cout<<"alpha:"<<alpha<<" kappa:"<<kappa<<std::endl;
        step.t().raw_print(std::cout,"step:");
        theta.t().raw_print(std::cout,"theta:");
        full_step.t().raw_print(std::cout,"fullstep:");
        lbound.t().raw_print(std::cout,"lbound:");
        ubound.t().raw_print(std::cout,"ubound:");
        throw NumericalError("bound_step: Infeasible Bounding failed upper bounds");
    }
    return kappa*alpha*step;
}

void compute_bound_scaling_vec(const VecT &theta, const VecT &g, const VecT &lbound, const VecT &ubound,
                              VecT &v, VecT &Jv)
{
    auto N = theta.n_elem;
    v.set_size(N);
    Jv.set_size(N);
    for(arma::uword i=0; i<N; i++){
        if(g(i)>=0) {
            if(std::isfinite(lbound(i))){
                v(i) = theta(i)-lbound(i);
                Jv(i) = sgn(v(i));
            } else {
                v(i) = 1;
                Jv(i) = 0;
            }
        } else {
            if(std::isfinite(ubound(i))){
                v(i) = theta(i)-ubound(i);
                Jv(i) = sgn(v(i));
            } else {
                v(i) = -1;
                Jv(i) = 0;
            }
        }
    }
}

VecT compute_D_scale(const VecT &oldDscale, const VecT &grad2)
{
    return arma::clamp(arma::max(oldDscale,arma::sqrt(arma::abs(grad2))), min_scaling,max_scaling);
}

void compute_scaled_problem(const MatT &H, const VecT &g, const VecT &Dinv, const VecT &Jv, MatT& Hhat, VecT &ghat)
{
    Hhat = arma::diagmat(Dinv) * -arma::symmatu(H) * arma::diagmat(Dinv) + arma::diagmat(-g % Jv);
    ghat = Dinv % -g;
}

double compute_initial_trust_radius(const VecT &ghat)
{
    return clamp(arma::norm(ghat), trust_radius_init_min, trust_radius_init_max);
}

VecT compute_cauchy_point(const VecT &g, const MatT &H, double delta)
{
    double gnorm = arma::norm(g);
    double Q = arma::dot(g,H*g);
    double tau = (Q<=0) ? 1 : std::min(1.0, gnorm*gnorm*gnorm / (delta*Q));
    return  -(tau*delta/gnorm) * g;
}

double compute_quadratic_model_value(const VecT &s, const VecT &g, const MatT &H)
{
    return arma::dot(g,s) + .5*arma::dot(s, arma::symmatu(H)*s);
}


VecT solve_TR_subproblem(const VecT &g, const MatT &H, double delta)
{
    auto N = g.n_elem;
    double g_norm = arma::norm(g);
    MatT Hchol = H;
    bool pos_def = cholesky(Hchol);

    if(pos_def) {
        VecT newton_step = cholesky_solve(Hchol,-g);  //this is s(0), i.e., lambda=0
        if(arma::norm(newton_step)<=delta){
            //[Case 1]: Full Newton Step
            return newton_step;
        } else {
            //[Case 2]: Restricted Newton Step
            //Attempt to arrive at bounds that avoids having to do an eigendecomposition in this case
            double lambda_lb = 0;
            double lambda_ub = g_norm/delta;
            return solve_restricted_step_length_newton(g,H,delta,lambda_lb, lambda_ub);
        }
    } else {
        //Indefinite hessian.  Do eigendecomposition to better understand the hessian
        VecT lambda_H;
        MatT Q_H;
        bool decomp_success = arma::eig_sym(lambda_H,Q_H, arma::symmatu(H)); //Compute eigendecomposition of symmetric matrix H
        if(!decomp_success) throw NumericalError("Could not eigendecompose");
        VecT g_hat = Q_H.t() * g; // g in coordinates of H's eigensystem
        double delta2 = delta * delta; //delta^2
        double lambda_min = lambda_H(0); //Minimum eigenvalue.  lambda_H is guaranteed in decreasing order

        // compute multiplicity of lambda_min.
        arma::uword Nmin = 1;
        while(lambda_H(Nmin)==lambda_min) Nmin++;
        //Compute ||P_min g'||^2
        double g_min_norm = 0.;
        for(arma::uword i=0; i<Nmin;i++) g_min_norm += g_hat(i)*g_hat(i);
        g_min_norm = sqrt(g_min_norm);
        if(g_min_norm>0) {
            //[Case 3]: Indefinite hessian, general-position gradient
            //The gradient extends into the lambda min subspace, so there is a pole at s(lambda_min)=inf
            double lambda_lb = g_min_norm/delta - lambda_min;
            double lambda_ub = g_norm/delta - lambda_min;
            return solve_restricted_step_length_newton(g,H,delta,lambda_lb, lambda_ub);
        } else {
            //Compute s_perp_sq = ||P^perp_min s(lambda_min)||^2
            VecT  s_perp_lmin(N,arma::fill::zeros);
            for(arma::uword i=Nmin; i<N; i++) s_perp_lmin += (g_hat(i)/(lambda_H(i)-lambda_min)) * Q_H.col(i);
            double s_perp_lmin_normsq = arma::dot(s_perp_lmin,s_perp_lmin);
            std::cout<<" s_perp_lmin_normsq: "<<s_perp_lmin_normsq<<"\n";
            if(s_perp_lmin_normsq >= delta) {
                //[Case 4]: Indefinite hessian, degenerate gradient, sufficient perpendicular step
                double lambda_lb = -lambda_min;
                double lambda_ub = g_norm/delta - lambda_min;
                std::cout<<" {{Case 4:  Indefinite hessian, degenerate gradient, sufficient perpendicular step}}";
                std::cout<<"  [lambda range: "<<lambda_lb<<"-"<<lambda_ub<<"]\n";
                return solve_restricted_step_length_newton(g,H,delta,lambda_lb, lambda_ub);
            } else {
                //[Case 5]: Indefinite hessian, degenerate gradient, insufficient perpendicular step
                //(i.e., The hard-hard case)
                double tau = sqrt(delta2 - s_perp_lmin_normsq);
                std::cout<<" {{Case 5:  Indefinite hessian, degenerate gradient, insufficient perpendicular step}}";
                std::cout<<" [tau: "<<tau<<" ] q1:"<<Q_H.col(0).t();
                return s_perp_lmin + tau * Q_H.col(0);  // s_perp(lambda_min) + tau * q_min; q_min is vector from S_min subspace
            }
        }
    }
}

VecT solve_restricted_step_length_newton(const VecT &g, const MatT &H, double delta, double lambda_lb, double lambda_ub)
{
    //Initially Hchol is the Cholesky decomposition at lambda=0
    double lambda = clamp(std::max(1./delta, .5*(lambda_lb+lambda_ub)), lambda_lb, lambda_ub);
    int max_iter = 50;

    for(int i=0; i<max_iter;i++) {
        MatT R = H;
        R.diag() += lambda;
        bool is_pos = cholesky(R);
        if(!is_pos) {
            throw NumericalError("Bad Cholesky decomposition.  Lambda is too small??.");
        }
        VecT p = cholesky_solve(R,-g);
        MatT Rtri = R;
        cholesky_convert_lower_triangular(Rtri);
        VecT q = arma::solve(arma::trimatl(Rtri), p);
        double norm_p = arma::norm(p);
        double objective = 1/delta - 1/norm_p;

        if(std::fabs(objective) < tr_subproblem_newton_epsilon) return p;
        double lambda_delta = (norm_p/arma::norm(q)) * (norm_p/arma::norm(q)) * ((norm_p-delta)/delta);
        if(std::fabs(lambda_delta) < tr_subproblem_newton_epsilon) return p;
        if(lambda==lambda_lb && lambda_delta<0) {
            throw NumericalError("Bad lambda lower bound??");
        }
        if(lambda==lambda_ub && lambda_delta>0) {
            throw NumericalError("Bad lambda upper bound??");
        }
        lambda += lambda_delta;
        lambda = std::min(std::max(lambda,lambda_lb),lambda_ub);
    }
    throw NumericalError("Lambda search exceeded max_iter");
}

} /* namespace mappel::estimator::subroutine */
} /* namespace mappel::estimator */
} /* namespace mappel */
