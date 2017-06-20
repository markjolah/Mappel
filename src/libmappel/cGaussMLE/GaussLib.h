/*!
 * \file GaussLib.h
 * \author Keith Lidke
 * \date January 10, 2010
 * \brief Numerical functions for computing gauss MLE function and derivative values.
 * These functions are used for the MLE fitting of model data to the various MLE models.
 */

#ifndef _GAUSSLIB_H
#define _GAUSSLIB_H

float IntGauss1D(int i, float mu, float sigma);

float alpha( float z,  float Ax,  float Bx,  float d);

float dalphadz( float z,  float Ax,  float Bx,  float d);

float d2alphadz2( float z,  float Ax,  float Bx,  float d);

void DerivativeIntGauss1D( int ii,  float x,  float sigma, 
         float N,  float PSFy, float *dudt, float *d2udt2);

void DerivativeIntGauss1DSigma( int ii,  float x,  float Sx, 
         float N,  float PSFy, float *dudt, float *d2udt2);

void DerivativeIntGauss2DSigma( int ii,  int jj, 
         float x,  float y,  float S,  float N, 
         float PSFx,  float PSFy, float *dudt, float *d2udt2);

void DerivativeIntGauss2Dz( int ii,  int jj,  float *theta,
         float PSFSigma_x,  float PSFSigma_y,  float Ax, 
         float Ay,  float Bx,  float By,  float gamma, 
         float d, float *pPSFx, float *pPSFy, float *dudt, float *d2udt2);

void CenterofMass2D( int sz,  const float *data, float *x, float *y);

void GaussFMaxMin2D( int sz,  float sigma,  const float * data,
                    float *MaxN, float *MinBG);

#endif /* _GAUSSLIB_H */
