# MAPPEL

Mappel is an object-oriented image processing library for [super-resolution localization](https://en.wikipedia.org/wiki/Super-resolution_microscopy#Localization_microscopy) of Gaussian point emitters in [fluorescence microscopy](https://en.wikipedia.org/wiki/Fluorescence_microscope#Sub-diffraction_techniques) applications.
* Mappel uses CMake and builds cross-platform for Linux  and Windows 64-bit.
* Mappel has object-oriented interfaces in C++, Python, and Matlab
* Mappel uses OpenMP to parallelize operations over vectors of images or parameters
* Mappel is free-as-in-beer and free-as-in-speech! ([Apache-2.0](LICENSE))

## Documentation
The Mappel Doxygen documentation can be build with the `OPT_DOC` CMake option and is also available on online:
  * [Mappel HTML Manual](https://markjolah.github.io/Mappel/index.html)
  * [Mappel PDF Manual](https://markjolah.github.io/Mappel/pdf/Mappel-0.0.3-reference.pdf)
  * [Mappel github repository](https://github.com/markjolah/Mappel)


## Background

Point emitter localization is a process of precisely estimating the sub-pixel location of a single point source emitter (molecule/protein) at effective resolutions 10-50 times smaller than the fundamental diffraction limit for optical microscopes.  Operationally, this is the process of going from the blurry, noisy, pixelated images to the estimate of true emitter position and the estimate of the uncertainty in  true position.  Figure 1 shows the point emitter localization process visually utilizing realistic physical scales for a typical super-resolution fluorescence microscope configuration, and showing typical effective fitting resolution.

<p align="center"><img alt="Fig 1: Effective fitting resolution in typical applications" src="https://raw.githubusercontent.com/markjolah/Mappel/master/doc/images/fitting_resolution.png" width="550"/>

<p align="center">
<strong>Figure 1</strong>: Effective fitting resolution in typical applications
</p>
</p>






## Installation
Mappel uses the CMake build system, and is designed to be cross-compiled from linux to other platforms, primarily Win64, although future OSX support is planned.


## Dependencies

Several standard numerical packages are required to build Mappel.  Most distributions should have development versions of these packages which provide the include files and
other necessary development files for the packages.

* [*Armadillo*](http://arma.sourceforge.net/docs.html) - A high-performance array library for C++.
* [Boost](http://www.boost.org/)
* BLAS
    * Requires support for 64-bit integers.
    * [Netlib BLAS Reference](http://www.netlib.org/blas/)
* LAPACK
    * Requires support for 64-bit integers.
    * [Netlib LAPACK Reference](http://www.netlib.org/lapack/)

Note the `OPT_BLAS_INT64` CMake option controls whether Armadillo uses BLAS and LAPACK libraries that use 64-bit interger indexing.
Matlab uses 64-bit by default, so linking Mappel to Matlab MEX libraries requires this option enabled.  Many linux systems only provide 32-bit integer versions of BLAS and Lapack, and the option can be disabled if Matlab support is not a concern and 64-bit support is difficult to provide.

### External Projects
These packages are specialized CMake projects.  If they are not currently installed on the development machines we use the [AddExternalDependency.cmake](https://github.com/markjolah/UncommonCMakeModules/blob/master/AddExternalDependency.cmake) which will automatically download, configure, build and install to the `CMAKE_INSTALL_PREFIX`, enabling their use through the normal CMake `find_package()` system.

- [BacktraceException](https://markjolah.github.iom/BacktraceException) - A library to provide debugging output
    on exception calls.  Important for Matlab debugging.
- [ParallelRngManager](https://markjolah.github.io/ParallelRngManager) -  A simple manager for easily deploying a set of RNG
   parallelized over a set number of threads, using the TRNG parallel RNG library.
- [PriorHessian](https:///markjolah.github.io/ParallelRngManager) - The PriorHessian library allows fast
    computation of log-likelihood and derivatives for composite priors.



# Model classes

Mappel provides model objects that correspond to different fitting-modes (psf-models).  Mappel's core is a C++ library `libmappel.so` that uses OpenMP to automatically parallelize localizations over multiple images.  Mappel also provides detailed object-oriented interfaces for Python and Matlab, using the same concept of a Model class to represent each class of psf fitting models.

## Computations available

 * `llh` - log-likelihood (log of pdf)
 * `rllh` - relative log-likelihood (log of pdf without constant terms)
 * `grad` - derivative of log-likelihood (or equivalently of relative-llh)
 * `grad2` - 2nd-derivative of log-likelihood
 * `hessian` - hessian of log-likelihood

# Design Notes
## Static Polymorphism

The Mappel library is designed using static polymorphism (templates), and as such avoids virtual functions for small-grained  tasks, and instead uses templates, which allow many small functions to be inlined.  This aggressive inlining by the compiler produces log-likelihood, gradient, and hessian functions that are nearly as fast as hand-coded functions.
