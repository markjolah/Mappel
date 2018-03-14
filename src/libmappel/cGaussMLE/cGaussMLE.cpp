/*!
* \file cGaussMLE.cpp
* \author Keith Lidke
* \date January 10, 2010
* \brief The MLE optimization routines 
*/
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
#include <cmath>

#include "definitions.h"
#include "matinv.h"
#include "GaussLib.h"
#include "cGaussMLE.h"

namespace mappel {
namespace cgauss {

/** @brief CGauss Initialization for 4-parameter model {x,y,I,bg}
 * @param data array of subregions to fit copied to GPU
 * @param PSFSigma sigma of the point spread function
 * @param size nxn size of the subregion to fit
 * @param[out] theta estimated theta must be size 4
 */
void MLEInit(const float data[], float PSFSigma, int size, float theta_est[])
{
    float Nmax;
    CenterofMass2D(size, data, &theta_est[0], &theta_est[1]);
    GaussFMaxMin2D(size, PSFSigma, data, &Nmax, &theta_est[3]);
    theta_est[2] = std::max(0.0f, (Nmax-theta_est[3])*2*PI*PSFSigma*PSFSigma); 
}

/** Core fitting routing for 4-parameter model
 * 
 */
void MLEFit(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
            float theta_est[])
{
    float model_val, cf, df, data_val;
    float PSFy, PSFx;
    int NV=NV_P;  //The dimension  of the space (# of paramters in theta)
    float dudt[NV_P];
    float d2udt2[NV_P];
    float NR_Numerator[NV_P], NR_Denominator[NV_P];
    float theta[NV_P];
    float maxjump[NV_P]={1., 1., 100., 2.};
    float gamma[NV_P]={1., 1., .5, 1.};
    
    if(static_cast<int>(theta_init.n_elem) == NV) {
        for(int n=0;n<NV;n++) theta[n] = theta_init(n);
    } else {
        MLEInit(data,PSFSigma,size,theta);
    }
    
    for (int kk=0;kk<iterations;kk++) {//main iterative loop
        //initialize
        memset(NR_Numerator,0,NV_P*sizeof(float));
        memset(NR_Denominator,0,NV_P*sizeof(float));
        
        for (int ii=0;ii<size;ii++) for(int jj=0;jj<size;jj++) {
            PSFx=IntGauss1D(ii, theta[0], PSFSigma);
            PSFy=IntGauss1D(jj, theta[1], PSFSigma);
            
            model_val=theta[3]+theta[2]*PSFx*PSFy;
            data_val=data[size*jj+ii];
            
            //calculating derivatives
            DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
            DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.;
            dudt[3] = 1.;
            d2udt2[3] = 0.;
            
            cf=0.;
            df=0.;
            if (model_val>10e-3f) cf=data_val/model_val-1;
            if (model_val>10e-3f) df=data_val/pow(model_val, 2);
            cf=std::min(cf, 10e4f);
            df=std::min(df, 10e4f);
            
            for (int ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        float step;
        if (kk<2) {
            for (int ll=0;ll<NV;ll++) {
                step=-gamma[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        } else {
            for (int ll=0;ll<NV;ll++) {
                step=-std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        }
        
        // Any other constraints
        theta[2]=std::max(theta[2], 1e0f);
        theta[3]=std::max(theta[3], 1e-2f);
    }
    for (int kk=0;kk<NV;kk++) theta_est[kk]=theta[kk];
}

void MLEFit_debug(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
                  float theta_est[], MatT &sequence)
{
    float model_val, cf, df, data_val;
    float PSFy, PSFx;
    int NV=NV_P;  //The dimension  of the space (# of paramters in theta)
    float dudt[NV_P];
    float d2udt2[NV_P];
    float NR_Numerator[NV_P], NR_Denominator[NV_P];
    float theta[NV_P];
    float maxjump[NV_P]={1., 1., 100., 2.};
    float gamma[NV_P]={1., 1., .5, 1.};
    sequence.set_size(NV,iterations+1);
    
    if(static_cast<int>(theta_init.n_elem) == NV) {
        for(int n=0;n<NV;n++) theta[n] = theta_init(n);
    } else {
        MLEInit(data,PSFSigma,size,theta);
    }
    for(int n=0;n<NV;n++) sequence(n,0) = theta[n]; //Record theta
    for (int kk=0;kk<iterations;kk++) {//main iterative loop
        //initialize
        memset(NR_Numerator,0,NV_P*sizeof(float));
        memset(NR_Denominator,0,NV_P*sizeof(float));
        
        for (int ii=0;ii<size;ii++) for(int jj=0;jj<size;jj++) {
            PSFx=IntGauss1D(ii, theta[0], PSFSigma);
            PSFy=IntGauss1D(jj, theta[1], PSFSigma);
            
            model_val=theta[3]+theta[2]*PSFx*PSFy;
            data_val=data[size*jj+ii];
            
            //calculating derivatives
            DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
            DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.;
            dudt[3] = 1.;
            d2udt2[3] = 0.;
            
            cf=0.;
            df=0.;
            if (model_val>10e-3f) cf=data_val/model_val-1;
            if (model_val>10e-3f) df=data_val/pow(model_val, 2);
            cf=std::min(cf, 10e4f);
            df=std::min(df, 10e4f);
            
            for (int ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        float step;
        if (kk<2) {
            for (int ll=0;ll<NV;ll++) {
                step=-gamma[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        } else {
            for (int ll=0;ll<NV;ll++) {
                step=-std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        }
        
        // Any other constraints
        theta[2]=std::max(theta[2], 1e0f);
        theta[3]=std::max(theta[3], 1e-2f);
        for(int n=0;n<NV;n++) sequence(n,kk+1) = theta[n]; //Record theta
    }
    for (int kk=0;kk<NV;kk++) theta_est[kk]=theta[kk];
}


/*! 
* \brief basic MLE fitting kernel.  No additional parameters are computed.
* \param data array of subregions to fit copied to GPU
* \param PSFSigma sigma of the point spread function
* \param size nxn size of the subregion to fit
* \param iterations number of iterations to run
* \param theta_est array of fitting parameters to return (theta)
* \param CRLB array of Cramer-Rao lower bound estimates to return for each parameter
* \param LogLikelihood loglikelihood estimates to return
* \param Nfits number of subregions to fit
*  theta are: {x,y,N,bg}
*/
void MLEFitFull(const float data[], const float PSFSigma, const int size,
        const int iterations, const FVecT &theta_init, float theta_est[], float CRLB[],
        float *LogLikelihood)
{
    float M[NV_P*NV_P], Diag[NV_P], Minv[NV_P*NV_P];
    int ii, jj, kk, ll;
    float model_val, cf, df, data_val;
    float Div;
    float PSFy, PSFx;
    int NV=NV_P;  //The dimension  of the space (# of paramters in theta)
    float dudt[NV_P];
    float d2udt2[NV_P];
    float NR_Numerator[NV_P], NR_Denominator[NV_P];
    float theta[NV_P];
    float maxjump[NV_P]={1., 1., 100., 2.};
    float gamma[NV_P]={1., 1., .5, 1.};

    memset(M,0,NV_P*NV_P*sizeof(float));
    memset(Minv,0,NV_P*NV_P*sizeof(float));
    
    //initial values
    if(!theta_init.is_empty() && theta_init.n_elem==4) {
        for(int n=0;n<4;n++) theta[n]=theta_init(n);
    } else {
        MLEInit(data,PSFSigma,size,theta);
    }
    
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        //initialize
        memset(NR_Numerator,0,NV_P*sizeof(float));
        memset(NR_Denominator,0,NV_P*sizeof(float));

        for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
            PSFx=IntGauss1D(ii, theta[0], PSFSigma);
            PSFy=IntGauss1D(jj, theta[1], PSFSigma);
            
            model_val=theta[3]+theta[2]*PSFx*PSFy;
            data_val=data[size*jj+ii];
            
            //calculating derivatives
            DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
            DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.;
            dudt[3] = 1.;
            d2udt2[3] = 0.;
            
            cf=0.;
            df=0.;
            if (model_val>10e-3f) cf=data_val/model_val-1;
            if (model_val>10e-3f) df=data_val/pow(model_val, 2);
            cf=std::min(cf, 10e4f);
            df=std::min(df, 10e4f);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        float step;
        if (kk<2) {
            for (ll=0;ll<NV;ll++) {
                step=-gamma[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        } else {
            for (ll=0;ll<NV;ll++) {
                step=-std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        }

        // Any other constraints
        theta[2]=std::max(theta[2], 1e0f);
        theta[3]=std::max(theta[3], 1e-2f);
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.;
    for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
        PSFx=IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=IntGauss1D(jj, theta[1], PSFSigma);
        
        model_val=theta[3]+theta[2]*PSFx*PSFy;
        data_val=data[size*jj+ii];
        
        //calculating derivatives
        DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
        DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.;
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model_val;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model_val>0){
            if (data_val>0) {
                Div+=data_val*log(model_val)-model_val-data_val*log(data_val)+data_val;
            } else {
                Div+=-model_val;
            }
        }
    }

    // Matrix inverse (CRLB=F^-1) and output assigments
    MatInvN(M, Minv, Diag, NV);
    
    //write to global arrays
    for (kk=0;kk<NV;kk++) theta_est[kk]=theta[kk];
    for (kk=0;kk<NV;kk++) CRLB[kk]=Diag[kk];
    *LogLikelihood = Div;
    
    return;
}


/** @brief CGauss Initialization for 5-parameter model {x,y,I,bg,sigma}
 * @param data array of subregions to fit copied to GPU
 * @param PSFSigma sigma of the point spread function
 * @param size nxn size of the subregion to fit
 * @param[out] theta estimated theta must be size 4
 */
void MLEInit_sigma(const float data[], float PSFSigma, int size, float theta_est[])
{
    //initial values
    float Nmax;
    CenterofMass2D(size, data, &theta_est[0], &theta_est[1]);
    GaussFMaxMin2D(size, PSFSigma, data, &Nmax, &theta_est[3]);
    theta_est[2] = std::max(0.0f, (Nmax-theta_est[3])*2*PI*PSFSigma*PSFSigma);
    theta_est[4] = PSFSigma;
}

void MLEFit_sigma(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
                  float theta_est[])
{
    /*!
    * \brief basic MLE fitting kernel.  No additional parameters are computed.
    * \param data array of subregions to fit copied to GPU
    * \param PSFSigma sigma of the point spread function
    * \param size nxn size of the subregion to fit
    * \param iterations number of iterations for solution to converge
    * \param theta_est array of fitting parameters to return for each subregion
    * \param CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
    * \param LogLikelihood array of loglikelihood estimates to return for each subregion
    * \param Nfits number of subregions to fit
    */

    //__shared__ float s_data[MEM];
    float model, cf, df, data_val;
    float PSFy, PSFx;
    int NV=NV_PS;
    float dudt[NV_PS];
    float d2udt2[NV_PS];
    float NR_Numerator[NV_PS], NR_Denominator[NV_PS];
    float theta[NV_PS];
    float maxjump[NV_PS]={1e0f, 1e0f, 1e2f, 2e0f, 5e-1f};
    float gamma[NV_PS]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f};

    if(static_cast<int>(theta_init.n_elem) == NV) {
        for(int n=0;n<NV;n++) theta[n] = theta_init(n);
    } else {
        MLEInit_sigma(data,PSFSigma,size,theta);
    }

    for (int kk=0;kk<iterations;kk++) {//main iterative loop
        //initialize
        memset(NR_Numerator,0,NV_PS*sizeof(float));
        memset(NR_Denominator,0,NV_PS*sizeof(float));

        for (int ii=0;ii<size;ii++) for(int jj=0;jj<size;jj++) {
            PSFx=IntGauss1D(ii, theta[0], theta[4]);
            PSFy=IntGauss1D(jj, theta[1], theta[4]);

            model=theta[3]+theta[2]*PSFx*PSFy;
            data_val=data[size*jj+ii];

            //calculating derivatives
            DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], &d2udt2[4]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0f;
            dudt[3] = 1.0f;
            d2udt2[3] = 0.0f;

            cf=0.0f;
            df=0.0f;
            if (model>10e-3f) cf=data_val/model-1;
            if (model>10e-3f) df=data_val/pow(model, 2);
            cf=std::min(cf, 10e4f);
            df=std::min(df, 10e4f);

            for (int ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }

        // The update
        float step;
        if (kk<5) {
            for (int ll=0;ll<NV;ll++){
                step=-gamma[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        } else {
            for (int ll=0;ll<NV;ll++) {
                step=-std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        }
        // Any other constraints
        theta[2]=std::max(theta[2], 1.0f);
        theta[3]=std::max(theta[3], 0.01f);
        theta[4]=std::max(theta[4], 0.5f);
        theta[4]=std::min(theta[4], size/2.0f);
    }
    for (int kk=0;kk<NV;kk++) theta_est[kk]=theta[kk];
}

void MLEFit_sigma_debug(const float data[], float PSFSigma, int size, int iterations, const FVecT &theta_init, 
                        float theta_est[], MatT &sequence)
{
    /*!
    * \brief basic MLE fitting kernel.  No additional parameters are computed.
    * \param data array of subregions to fit copied to GPU
    * \param PSFSigma sigma of the point spread function
    * \param size nxn size of the subregion to fit
    * \param iterations number of iterations for solution to converge
    * \param theta_est array of fitting parameters to return for each subregion
    * \param CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
    * \param LogLikelihood array of loglikelihood estimates to return for each subregion
    * \param Nfits number of subregions to fit
    */

    //__shared__ float s_data[MEM];
    float model, cf, df, data_val;
    float PSFy, PSFx;
    int NV=NV_PS;
    float dudt[NV_PS];
    float d2udt2[NV_PS];
    float NR_Numerator[NV_PS], NR_Denominator[NV_PS];
    float theta[NV_PS];
    float maxjump[NV_PS]={1e0f, 1e0f, 1e2f, 2e0f, 5e-1f};
    float gamma[NV_PS]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f};
    sequence.set_size(NV,iterations+1);

    if(static_cast<int>(theta_init.n_elem) == NV) {
        for(int n=0;n<NV;n++) theta[n] = theta_init(n);
    } else {
        MLEInit_sigma(data,PSFSigma,size,theta);
    }
    for(int n=0;n<NV;n++) sequence(n,0) = theta[n]; //Record theta
    for (int kk=0;kk<iterations;kk++) {//main iterative loop
        //initialize
        memset(NR_Numerator,0,NV_PS*sizeof(float));
        memset(NR_Denominator,0,NV_PS*sizeof(float));

        for (int ii=0;ii<size;ii++) for(int jj=0;jj<size;jj++) {
            PSFx=IntGauss1D(ii, theta[0], theta[4]);
            PSFy=IntGauss1D(jj, theta[1], theta[4]);

            model=theta[3]+theta[2]*PSFx*PSFy;
            data_val=data[size*jj+ii];

            //calculating derivatives
            DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], &d2udt2[4]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0f;
            dudt[3] = 1.0f;
            d2udt2[3] = 0.0f;

            cf=0.0f;
            df=0.0f;
            if (model>10e-3f) cf=data_val/model-1;
            if (model>10e-3f) df=data_val/pow(model, 2);
            cf=std::min(cf, 10e4f);
            df=std::min(df, 10e4f);

            for (int ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }

        // The update
        float step;
        if (kk<5) {
            for (int ll=0;ll<NV;ll++){
                step=-gamma[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        } else {
            for (int ll=0;ll<NV;ll++) {
                step=-std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        }
        // Any other constraints
        theta[2]=std::max(theta[2], 1.0f);
        theta[3]=std::max(theta[3], 0.01f);
        theta[4]=std::max(theta[4], 0.5f);
        theta[4]=std::min(theta[4], size/2.0f);
        for(int n=0;n<NV;n++) sequence(n,kk+1)=theta[n]; //Record theta
    }
    for (int kk=0;kk<NV;kk++) theta_est[kk]=theta[kk];
}


void MLEFit_sigma_full(const float data[], const float PSFSigma, const int size, const int iterations,
                  float theta_est[], float CRLB[], float *LogLikelihood){
    /*!
    * \brief basic MLE fitting kernel.  No additional parameters are computed.
    * \param data array of subregions to fit copied to GPU
    * \param PSFSigma sigma of the point spread function
    * \param size nxn size of the subregion to fit
    * \param iterations number of iterations for solution to converge
    * \param theta_est array of fitting parameters to return for each subregion
    * \param CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
    * \param LogLikelihood array of loglikelihood estimates to return for each subregion
    * \param Nfits number of subregions to fit
    */

    //__shared__ float s_data[MEM];
    float M[NV_PS*NV_PS], Diag[NV_PS], Minv[NV_PS*NV_PS];
    int ii, jj, kk, ll;
    float model, cf, df, data_val;
    float Div;
    float PSFy, PSFx;
    int NV=NV_PS;
    float dudt[NV_PS];
    float d2udt2[NV_PS];
    float NR_Numerator[NV_PS], NR_Denominator[NV_PS];
    float theta[NV_PS];
    float maxjump[NV_PS]={1e0f, 1e0f, 1e2f, 2e0f, 5e-1f};
    float gamma[NV_PS]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f};
    float Nmax;


    memset(M,0,NV_PS*NV_PS*sizeof(float));
    memset(Minv,0,NV_PS*NV_PS*sizeof(float));

    //initial values
    CenterofMass2D(size, data, &theta[0], &theta[1]);
    GaussFMaxMin2D(size, PSFSigma, data, &Nmax, &theta[3]);
    theta[2]=std::max(0.0f, (Nmax-theta[3])*2*PI*PSFSigma*PSFSigma);
    theta[4]=PSFSigma;

    for (kk=0;kk<iterations;kk++) {//main iterative loop

        //initialize
        memset(NR_Numerator,0,NV_PS*sizeof(float));
        memset(NR_Denominator,0,NV_PS*sizeof(float));

        for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
            PSFx=IntGauss1D(ii, theta[0], theta[4]);
            PSFy=IntGauss1D(jj, theta[1], theta[4]);

            model=theta[3]+theta[2]*PSFx*PSFy;
            data_val=data[size*jj+ii];

            //calculating derivatives
            DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], &d2udt2[4]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0f;
            dudt[3] = 1.0f;
            d2udt2[3] = 0.0f;

            cf=0.0f;
            df=0.0f;
            if (model>10e-3f) cf=data_val/model-1;
            if (model>10e-3f) df=data_val/pow(model, 2);
            cf=std::min(cf, 10e4f);
            df=std::min(df, 10e4f);

            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }

        // The update
        float step;
        if (kk<5) {
            for (ll=0;ll<NV;ll++){
                step=-gamma[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        } else {
            for (ll=0;ll<NV;ll++) {
                step=-std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
                theta[ll]+=step;
            }
        }
        // Any other constraints
        theta[2]=std::max(theta[2], 1.0f);
        theta[3]=std::max(theta[3], 0.01f);
        theta[4]=std::max(theta[4], 0.5f);
        theta[4]=std::min(theta[4], size/2.0f);
    }

    // Calculating the CRLB and LogLikelihood
    Div=0.0f;
    for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
        PSFx=IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=IntGauss1D(jj, theta[1], PSFSigma);

        model=theta[3]+theta[2]*PSFx*PSFy;
        data_val=data[size*jj+ii];

        //calculating derivatives
        DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], NULL);
        DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0f;

        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }

        //LogLikelyhood
        if (model>0) {
            if (data_val>0) {
                Div+=data_val*log(model)-model-data_val*log(data_val)+data_val;
            } else {
                Div+=-model;
            }
        }
    }

    // Matrix inverse (CRLB=F^-1) and output assigments
    MatInvN(M, Minv, Diag, NV);


    //write to global arrays
    for (kk=0;kk<NV;kk++) theta_est[kk]=theta[kk];
    for (kk=0;kk<NV;kk++) CRLB[kk]=Diag[kk];
    *LogLikelihood = Div;

    return;
}

// //*******************************************************************************************
// void kernel_MLEFit_z(const int subregion, const float *d_data, const float PSFSigma_x, const float Ax, const float Ay, const float Bx, 
//     const float By, const float gamma, const float d, const float PSFSigma_y, const int size, const int iterations, 
//         float *d_theta_est, float *d_CRLBs, float *d_LogLikelihood,const int Nfits){
//     /*! 
//     * \brief basic MLE fitting kernel.  No additional parameters are computed.
//     * \param d_data array of subregions to fit copied to GPU
//     * \param PSFSigma_x sigma of the point spread function on the x axis
//     * \param Ax ???
//     * \param Ay ???
//     * \param Bx ???
//     * \param By ???
//     * \param gamma ???
//     * \param d ???
//     * \param PSFSigma_y sigma of the point spread function on the y axis
//     * \param size nxn size of the subregion to fit
//     * \param iterations number of iterations for solution to converge
//     * \param d_theta_est array of fitting parameters to return for each subregion
//     * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
//     * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
//     * \param Nfits number of subregions to fit
//     */
//     //__shared__ float s_data[MEM];
//     float M[5*5], Diag[5], Minv[5*5];
//     //int tx = threadIdx.x;
//     //int bx = blockIdx.x;
//     //int BlockSize = blockDim.x;
//     int ii, jj, kk, ll;
//     float model, cf, df, data;
//     float Div;
//     float PSFy, PSFx;
//     int NV=5;
//     float dudt[5];
//     float d2udt2[5];
//     float NR_Numerator[5], NR_Denominator[5];
//     float theta[5];
//     float maxjump[5]={1e0f, 1e0f, 1e2f, 2e0f, 1e-1f};
//     float g[5]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f};
//     float Nmax;
//     
//     //Prevent read/write past end of array
//     if (subregion >= Nfits) return;
// 
//     memset(M,0,NV*NV*sizeof(float));
//     memset(Minv,0,NV*NV*sizeof(float));      
//     
//     //load data
//     const float *s_data = d_data+(size*size*subregion);
// 
//     //initial values
//     kernel_CenterofMass2D(size, s_data, &theta[0], &theta[1]);
//     kernel_GaussFMaxMin2D(size, PSFSigma_x, s_data, &Nmax, &theta[3]);
//     theta[2]=std::max(0.0f, (Nmax-theta[3])*2*PI*PSFSigma_x*PSFSigma_y*sqrt(2.0f));
//     theta[4]=0;
// 
//     for (kk=0;kk<iterations;kk++) {//main iterative loop
//         
//         //initialize
//         memset(NR_Numerator,0,NV*sizeof(float));
//         memset(NR_Denominator,0,NV*sizeof(float));
//         
//         for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
//             kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, dudt, d2udt2);
//             
//             model=theta[3]+theta[2]*PSFx*PSFy;
//             data=s_data[size*jj+ii];
//             
//             //calculating remaining derivatives
//             dudt[2] = PSFx*PSFy;
//             d2udt2[2] = 0.0f;
//             dudt[3] = 1.0f;
//             d2udt2[3] = 0.0f;
//             
//             cf=0.0f;
//             df=0.0f;
//             if (model>10e-3f) cf=data/model-1;
//             if (model>10e-3f) df=data/pow(model, 2);
//             cf=std::min(cf, 10e4f);
//             df=std::min(df, 10e4f);
//             
//             for (ll=0;ll<NV;ll++){
//                 NR_Numerator[ll]+=dudt[ll]*cf;
//                 NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
//             }
//         }
//         
//         // The update
//         if (kk<2)
//             for (ll=0;ll<NV;ll++)
//                 theta[ll]-=g[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
//         else
//             for (ll=0;ll<NV;ll++)
//                 theta[ll]-=std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
//         
//         // Any other constraints
//         theta[2]=std::max(theta[2], 1.0f);
//         theta[3]=std::max(theta[3], 0.01f);
//         
//     }
//     
//     // Calculating the CRLB and LogLikelihood
//     Div=0.0f;
//     for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
//         
//         kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay, Bx,By, gamma, d, &PSFx, &PSFy, dudt, NULL);
//         
//         model=theta[3]+theta[2]*PSFx*PSFy;
//         data=s_data[size*jj+ii];
//         
//         //calculating remaining derivatives
//         dudt[2] = PSFx*PSFy;
//         dudt[3] = 1.0f;
//     
//         //Building the Fisher Information Matrix
//         for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
//             M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
//             M[ll*NV+kk]=M[kk*NV+ll];
//         }
//         
//         //LogLikelyhood
//         if (model>0) {
//             if (data>0) { 
//                 Div+=data*log(model)-model-data*log(data)+data;
//             } else {
//                 Div+=-model;
//             }
//         }
//     }
//     
//     // Matrix inverse (CRLB=F^-1) 
//     kernel_MatInvN(M, Minv, Diag, NV);
// 
// //write to global arrays
//     for (kk=0;kk<NV;kk++) d_theta_est[Nfits*kk+subregion]=theta[kk];
//     for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
//     d_LogLikelihood[subregion] = Div;
//     return;
// }
// 
// //*******************************************************************************************
// void kernel_MLEFit_sigmaxy(const int subregion, const float *d_data, const float PSFSigma, const int size, const int iterations, 
//         float *d_theta_est, float *d_CRLBs, float *d_LogLikelihood,const int Nfits){
//     /*! 
//     * \brief basic MLE fitting kernel.  No additional parameters are computed.
//     * \param d_data array of subregions to fit copied to GPU
//     * \param PSFSigma sigma of the point spread function
//     * \param size nxn size of the subregion to fit
//     * \param iterations number of iterations for solution to converge
//     * \param d_theta_est array of fitting parameters to return for each subregion
//     * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
//     * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
//     * \param Nfits number of subregions to fit
//     */
// 
//     //__shared__ float s_data[MEM];
//     float M[6*6], Diag[6], Minv[6*6];
//     //int tx = threadIdx.x;
//     //int bx = blockIdx.x;
//     //int BlockSize = blockDim.x;
//     int ii, jj, kk, ll;
//     float model, cf, df, data;
//     float Div;
//     float PSFy, PSFx;
//     int NV=6;
//     float dudt[6];
//     float d2udt2[6];
//     float NR_Numerator[6], NR_Denominator[6];
//     float theta[6];
//     float maxjump[6]={1e0f, 1e0f, 1e2f, 2e0f, 1e-1f,1e-1f};
//     float g[6]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f,1.0f};
//     float Nmax;
//     
//     //Prevent read/write past end of array
//     if (subregion>=Nfits) return;
//     
//     memset(M,0,NV*NV*sizeof(float));
//     memset(Minv,0,NV*NV*sizeof(float));      
//     
//     //load data
//     const float *s_data = d_data+(size*size*subregion);
//     
//     //initial values
//     kernel_CenterofMass2D(size, s_data, &theta[0], &theta[1]);
//     kernel_GaussFMaxMin2D(size, PSFSigma, s_data, &Nmax, &theta[3]);
//     theta[2]=std::max(0.0f, (Nmax-theta[3])*2*PI*PSFSigma*PSFSigma);
//     theta[4]=PSFSigma;
//     theta[5]=PSFSigma;
//     for (kk=0;kk<iterations;kk++) {//main iterative loop
//         
//         //initialize
//         memset(NR_Numerator,0,NV*sizeof(float));
//         memset(NR_Denominator,0,NV*sizeof(float));
//         
//         for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
//             PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
//             PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);
//             
//             model=theta[3]+theta[2]*PSFx*PSFy;
//             data=s_data[size*jj+ii];
//             
//             //calculating derivatives
//             kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
//             kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], &d2udt2[1]);
//             kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], &d2udt2[4]);
//             kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], &d2udt2[5]);
//             dudt[2] = PSFx*PSFy;
//             d2udt2[2] = 0.0f;
//             dudt[3] = 1.0f;
//             d2udt2[3] = 0.0f;
//             
//             cf=0.0f;
//             df=0.0f;
//             if (model>10e-3f) cf=data/model-1;
//             if (model>10e-3f) df=data/pow(model, 2);
//             cf=std::min(cf, 10e4f);
//             df=std::min(df, 10e4f);
//             
//             for (ll=0;ll<NV;ll++){
//                 NR_Numerator[ll]+=dudt[ll]*cf;
//                 NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
//             }
//         }
//         
//         // The update
//             for (ll=0;ll<NV;ll++)
//                 theta[ll]-=g[ll]*std::min(std::max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
// 
//         // Any other constraints
//         theta[2]=std::max(theta[2], 1.0f);
//         theta[3]=std::max(theta[3], 0.01f);
//         theta[4]=std::max(theta[4], PSFSigma/10.0f);
//         theta[5]=std::max(theta[5], PSFSigma/10.0f);  
//     }
//     
//     // Calculating the CRLB and LogLikelihood
//     Div=0.0f;
//     for (ii=0;ii<size;ii++) for(jj=0;jj<size;jj++) {
//         
//         PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
//         PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);
//         
//         model=theta[3]+theta[2]*PSFx*PSFy;
//         data=s_data[size*jj+ii];
//         
//         //calculating derivatives
//         kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
//         kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], NULL);
//         kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], NULL);
//         kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], NULL);
//         dudt[2] = PSFx*PSFy;
//         dudt[3] = 1.0f;
//         
//         //Building the Fisher Information Matrix
//         for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
//             M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
//             M[ll*NV+kk]=M[kk*NV+ll];
//         }
//         
//         //LogLikelyhood
//         if (model>0) {
//             if (data>0) {
//                 Div+=data*log(model)-model-data*log(data)+data;
//             } else {
//                 Div+=-model;
//             }
//         }
//     }
//     
//     // Matrix inverse (CRLB=F^-1) and output assigments
//     kernel_MatInvN(M, Minv, Diag, NV);
// 
//     //write to global arrays
//     for (kk=0;kk<NV;kk++) d_theta_est[Nfits*kk+subregion]=theta[kk];
//     for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
//     d_LogLikelihood[subregion] = Div;
//     return;
// }

} /* namespace cgauss */
} /* namespace mappel */

