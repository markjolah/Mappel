/*!
 * \file cGaussMLE.h
 * \author Keith Lidke
 * \date January 10, 2010
 * \brief The MLE optimization routines 
 */

#ifndef _CGAUSSMLE_H
#define _CGAUSSMLE_H

void MLEFit(const float data[], const float PSFSigma, const int size,
            const int iterations,float Parameters[], float CRLB[],
            float *LogLikelihood);

void MLEFit_sigma(const float data[], const float PSFSigma, const int sz,
                  const int iterations, float Parameters[], float CRLBs[],
                  float *d_LogLikelihood);
// 
// void MLEFit_z(const float *d_data, const float PSFSigma_x, const float Ax, 
//               const float Ay, const float Bx, const float By, const float gamma,
//               const float d, const float PSFSigma_y, const int sz, 
//               const int iterations, float *d_Parameters, float *d_CRLBs, 
//               float *d_LogLikelihood);
// 
// void MLEFit_sigmaxy(const int subregion, const float *d_data, 
//                     const float PSFSigma, const int sz, const int iterations, 
//                     float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,
//                     const int Nfits);

#endif /* _CGAUSSMLE_H */
