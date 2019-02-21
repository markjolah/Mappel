% Package: Mappel
% Maximum a posteriori Point Emitter Localzatization
% Author: Mark J. Olah
% 2014-2019
% LICENSE: Apache-2.0.  See LICENSE file.
%
% Mappel is object-oriented.  Each fitting problem type has a specific class that is used to estimate
% parameters for its particular data and psf model.  The models include three primary components:
% Model point-spread-function shape
%   * Gaussian2D fixed sigma - A fixed Gaussian point-spread-function with shape PSFSigmaMin=[sigmaX, sigmaY] in pixels. 
%   * Gaussian2D linearly-variable-sigma - A linearly variable Gaussian point-spread-function with shape sigma_ratio*PSFSigmaMin in pixels. 
%   * Gaussian2D astigmaticly-variable-Sigma - A fully astigmatic Gaussian point-spread-function with bounds PSFSigmaMin, PSFSigmaMax in pixels. 
%  Model noise assumptions:
%   * PoissonNoise - Noise is purely poisson.  EMCCD with proper gain-calibrations and image normalization
%  Model estimator objective:
%   * Maximum a posteriori [MAP] - The MAP models include the prior likelihood in their estimation (optimization)
%   * Maximum likelihood estimation [MLE] - The MLE models use pure-likelihood with no prior information in their estimation (optimization)
%  Model dimension:
%   * 2D - Typical applications
%   * 1D - Used to reliably initialize (bootstrap) 2D estimators.
%
% Emitter Localization Classes:
%  * Mappel.MappelBase - Base class and most important class.  Provides the structure for all the emiiter
%     classes.  Also provides the C++ interface abstraction via MexIFace.MexIFaceMixin base class.
%  
% * Gaussian2D fixed sigma
%   * Mappel.Gauss2DMAP([sizeX,sizeY], [PSFSigmaX,PSFSigmaY])
%   * Mappel.Gauss2DMLE([sizeX,sizeY], [PSFSigmaX,PSFSigmaY])
% * Gaussian2D linearly-variable-sigma 
%   * Mappel.Gauss2DsMAP([sizeX,sizeY], [MinPSFSigmaX,MinPSFSigmaY], max_sigma_ratio)
%   * Mappel.Gauss2DsMLE([sizeX,sizeY], [MinPSFSigmaX,MinPSFSigmaY], max_sigma_ratio)
% * Gaussian2D astigmaticly-variable-Sigma
%   * Mappel.Gauss2DsxyMAP([sizeX,sizeY], [MinPSFSigmaX,MinPSFSigmaY], [MaxPSFSigmaX,MaxPSFSigmaY])
%   * Mappel.Gauss2DsxyMLE([sizeX,sizeY], [MinPSFSigmaX,MinPSFSigmaY], [MaxPSFSigmaX,MaxPSFSigmaY])
%
% See also Mappel.MappelBase Mappel.Gauss2DMAP Mappel.Gauss2DMLE Mappel.Gauss2DsMAP Mappel.Gauss2DsMLE
