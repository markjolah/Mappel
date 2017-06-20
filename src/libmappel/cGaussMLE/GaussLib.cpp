/*!
* \file GaussLib.cuh
* \author Keith Lidke
* \date January 10, 2010
* \brief Numerical functions for computing gauss MLE function and derivative values.
* These functions are used for the MLE fitting of model data to the various MLE models.
* 
* Mark J. Olah (mjo@cs.unm.edu) 
* 12-16-2013
* Modifications made to optimize for CPU performance.
*/
#include <cmath>

#include "definitions.h"
#include "GaussLib.h"

char d_assert[1024];

//	 /f$ \frac{1}{2}(erf((ii-x+frac{1}{2})*sqrt(\frac{1}{2}\sigma^2}))-erf((ii-x-frac{1}{2})*sqrt(\frac{1}{2}\sigma^2})))) /f$

//*******************************************************************************************
float IntGauss1D(int i, float mu, float sigma) {
    /*! 
    * \brief /f$ \frac{1}{2} /f$
    * \param i 
    * \param mu ???
    * \param sigma sigma value of the PSF
    * \return float
    * Integrates a gaussian with mean mu and sd sigma, from i-.5 to i+.5
    * /f$ \frac{1}{2}(erf((ii-x+frac{1}{2})*sqrt(\frac{1}{2}\sigma^2}))-erf((ii-x-frac{1}{2})*sqrt(\frac{1}{2}\sigma^2})))) /f$
    */
    float norm=1./(sigma*sqrt(2.));
    return 0.5*(erf(norm*(i+.5-mu))-erf(norm*(i-.5-mu)));
}


//*******************************************************************************************
float alpha(float z, float Ax, float Bx, float d) {
    /*! 
    * \brief compute coefficient for alpha
    * \param z ???
    * \param Ax ???
    * \param Bx ???
    * \param d ???
    * \return float alpha value
    */
    
    return 1.0f+pow(z/d, 2)+Ax*pow(z/d, 3)+Bx*pow(z/d, 4);
}

//*******************************************************************************************
float dalphadz( float z, float Ax, float Bx, float d){
    /*! 
    * \brief compute first derivative for alpha in relation to z
    * \param z ???
    * \param Ax ???
    * \param Bx ???
    * \param d ???
    * \return float alpha value
    */
    return (2.0f*z/(d*d) + 3.0f*Ax*pow(z, 2)/(d*d*d)+4.0f*Bx*pow(z, 3)/pow(d, 4));
}

//*******************************************************************************************
float d2alphadz2( float z, float Ax, float Bx, float d){
    /*! 
    * \brief compute second derivative for alpha in relation to z
    * \param z ???
    * \param Ax ???
    * \param Bx ???
    * \param d ???
    * \return float alpha value
    */
    return (2.0f/(d*d) + 6.0f*Ax*z/(d*d*d)+12.0f*Bx*pow(z, 2)/pow(d, 4));
}

//*******************************************************************************************
void DerivativeIntGauss1D(int i, float x, float sigma, float N,
         float PSFy, float dudt[], float *d2udt2) {
    /*! 
    * \brief compute the derivative of the 1D gaussian
    * \param i ???
    * \param x ???
    * \param sigma ???
    * \param N ???
    * \param PSFy ???
    * \param dudt ???
    * \param d2udt2 ???
    */    
    float a, b;
    
    a = exp(-.5*pow(((i+0.5f-x)/sigma), 2.0f));
    b = exp(-.5*pow((i-0.5f-x)/sigma, 2.0f));
    
    *dudt = -N/sqrt(2.0f*PI)/sigma*(a-b)*PSFy;
    
    if (d2udt2)
        *d2udt2 =-N/sqrt(2.0f*PI)/pow(sigma, 3)*((i+0.5f-x)*a-(i-0.5f-x)*b)*PSFy;
}

//*******************************************************************************************
void DerivativeIntGauss1DSigma( int ii,  float x, 
         float Sx,  float N,  float PSFy, float *dudt, float *d2udt2) {
    /*! 
    * \brief compute the derivative of the 1D gaussian
    * \param ii ???
    * \param x ???
    * \param Sx ???
    * \param N ???
    * \param PSFy ???
    * \param dudt ???
    * \param d2udt2 ???
    */    
    
    float ax, bx;
    
    ax = exp(-1.0f/2.0f*pow(((ii+0.5f-x)/Sx), 2.0f));
    bx = exp(-1.0f/2.0f*pow((ii-0.5f-x)/Sx, 2.0f)); 
    *dudt = -N/sqrt(2.0f*PI)/Sx/Sx*(ax*(ii-x+0.5f)-bx*(ii-x-0.5f))*PSFy;
    
    if (d2udt2)
        *d2udt2 =-2.0f/Sx*dudt[0]-N/sqrt(2.0f*PI)/pow(Sx, 5)*(ax*pow((ii-x+0.5f), 3)-bx*pow((ii-x-0.5f), 3))*PSFy;
}

//*******************************************************************************************
void DerivativeIntGauss2DSigma( int ii,  int jj,  float x,  float y,
         float S,  float N,  float PSFx,  float PSFy, float *dudt, float *d2udt2) {
    /*! 
    * \brief compute the derivative of the 2D gaussian
    * \param ii ???
    * \param jj ???
    * \param x ???
    * \param y ???
    * \param S ???
    * \param N ???
    * \param PSFx ???
    * \param PSFy ???
    * \param dudt ???
    * \param d2udt2 ???
    */    
    
    float dSx, dSy, ddSx, ddSy;
    
    DerivativeIntGauss1DSigma(ii,x,S,N,PSFy,&dSx,&ddSx);
    DerivativeIntGauss1DSigma(jj,y,S,N,PSFx,&dSy,&ddSy);

    *dudt    = dSx+dSy;
    if (d2udt2) *d2udt2 =ddSx+ddSy;
    
}

//*******************************************************************************************
void DerivativeIntGauss2Dz( int ii,  int jj,  float *theta,
         float PSFSigma_x,  float PSFSigma_y,  float Ax,  float Ay, 
         float Bx,  float By,  float gamma,  float d, float *pPSFx, float *pPSFy, float *dudt, float *d2udt2) {
    /*! 
    * \brief compute the derivative of the 2D gaussian
    * \param ii ???
    * \param jj ???
    * \param theta ???
    * \param PSFSigma_x ???
    * \param PSFSigma_y ???
    * \param Ax ???
    * \param Ay ???
    * \param Bx ???
    * \param By ???
    * \param gamma ???
    * \param d ???
    * \param pPSFx ???
    * \param pPSFy ???
    * \param dudt ???
    * \param d2udt2 ???
    */    
    
    float Sx, Sy, dSx, dSy, ddSx, ddSy, dSdzx, dSdzy,ddSddzx,ddSddzy;
    float z, PSFx, PSFy,alphax,alphay,ddx,ddy;
    float dSdalpha_x,dSdalpha_y,d2Sdalpha2_x,d2Sdalpha2_y;
    z=theta[4];
    
    alphax  = alpha(z-gamma, Ax, Bx, d);
    alphay  = alpha(z+gamma, Ay, By, d);

    Sx=PSFSigma_x*sqrt(alphax);
    Sy=PSFSigma_y*sqrt(alphay);
    
    PSFx=IntGauss1D(ii, theta[0], Sx);
    PSFy=IntGauss1D(jj, theta[1], Sy);
    *pPSFx=PSFx;
    *pPSFy=PSFy;
    
    DerivativeIntGauss1D(ii, theta[0], Sx, theta[2], PSFy, &dudt[0], &ddx);
    DerivativeIntGauss1D(jj, theta[1], Sy, theta[2], PSFx, &dudt[1], &ddy);
    DerivativeIntGauss1DSigma(ii, theta[0], Sx, theta[2], PSFy, &dSx, &ddSx);
    DerivativeIntGauss1DSigma(jj, theta[1], Sy, theta[2], PSFx, &dSy, &ddSy);

    dSdalpha_x=PSFSigma_x/2.0f/sqrt(alphax);
    dSdalpha_y=PSFSigma_y/2.0f/sqrt(alphay);
    
    dSdzx  = dSdalpha_x*dalphadz(z-gamma, Ax, Bx, d); 
    dSdzy  = dSdalpha_y*dalphadz(z+gamma, Ay, By, d);
    dudt[4] = dSx*dSdzx+dSy*dSdzy;
    
    if (d2udt2){
    d2udt2[0] =ddx;
    d2udt2[1] =ddy;
    
    d2Sdalpha2_x=-PSFSigma_x/4.0f/pow(alphax,1.5f);
    d2Sdalpha2_y=-PSFSigma_y/4.0f/pow(alphay,1.5f);
    
    ddSddzx  = d2Sdalpha2_x*pow(dalphadz(z-gamma, Ax, Bx, d),2)+dSdalpha_x*d2alphadz2(z-gamma, Ax, Bx, d); 
    ddSddzy  = d2Sdalpha2_y*pow(dalphadz(z+gamma, Ay, By, d),2)+dSdalpha_y*d2alphadz2(z+gamma, Ay, By, d); 
    
    d2udt2[4] =ddSx*(dSdzx*dSdzx)+dSx*ddSddzx+
            ddSy*(dSdzy*dSdzy)+dSy*ddSddzy;
    }
}

//*******************************************************************************************
void CenterofMass2D( int sz,  const float *data, float *x, float *y) {
    /*!
    * \brief compute the 2D center of mass of a subregion
    * \param sz nxn size of the subregion
    * \param data subregion to search
    * \param x x coordinate to return
    * \param y y coordinate to return
    */
    float tmpx=0.0f;
    float tmpy=0.0f;
    float tmpsum=0.0f;
    int ii, jj;
    
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        tmpx+=data[sz*jj+ii]*ii;
        tmpy+=data[sz*jj+ii]*jj;
        tmpsum+=data[sz*jj+ii];
    }
    *x=tmpx/tmpsum;
    *y=tmpy/tmpsum;
}

//*******************************************************************************************
void GaussFMaxMin2D( int sz,  float sigma,  const float * data, float *MaxN, float *MinBG) {
    /*!
    * \brief returns filtered min and pixels of a given subregion
    * \param sz nxn size of the subregion
    * \param sigma used in filter calculation
    * \param data the subregion to search
    * \param MaxN maximum pixel value
    * \param MinBG minimum background value
    */
    int ii, jj, kk, ll;
    float filteredpixel=0, sum=0;
    *MaxN=0.0f;
    *MinBG=10e10f; //big
    
    float norm=1.0f/2.0f/sigma/sigma;
    //loop over all pixels
    for (kk=0;kk<sz;kk++) for (ll=0;ll<sz;ll++){
        filteredpixel=0.0f;
        sum=0.0f;
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++){
            filteredpixel+=exp(-pow((float)(ii-kk), 2)*norm)*exp(-pow((float)(ll-jj), 2)*norm)*data[ii*sz+jj];
            sum+=exp(-pow((float)(ii-kk), 2)*norm)*exp(-pow((float)(ll-jj), 2)*norm);
        }
        filteredpixel/=sum;
        
        *MaxN=max(*MaxN, filteredpixel);
        *MinBG=min(*MinBG, filteredpixel);
    }
}












