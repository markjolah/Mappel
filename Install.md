[Installation]


The following Environment variables control various aspects of the install:


[Dependecies]

Locations of dependencies [optional] Default to use the HEAD version from github

ParallelRngManagerURL - Local directory or git URL for the ParallelRngManager library 
PriorHessianURL - Local directory or git URL for the PriorHessian library 

[Debugging]

Additional dependency

BacktraceExceptionURL - Local directory or git URL for the BacktraceException library [optional] Default to use the HEAD version from github

[Matlab support]

There is a Cmake option "Matlab" that controls Matlab support.

MexIFaceURL- Local directory or git URL for the MexIface library (MATLAB Support). [optional] Default to use the HEAD version from github
MATLAB_LIBS_ROOT - [Optional] Local path to find Matlab core shared libraries to link against (overrides default search paths).
                   Must constain subdirectory structure ($MATLAB_ARCH)/$(MATLAB_VERSION)/{bin,extern}.  MATLAB_ARCH is [glnxa64, maci64, win64].
MATLAB_ROOT_GLNXA64 - Necessary for Matlab.  Location of the matlab WIN64 version to link against.

[Cross-building to Win64]

MXE_ROOT- Local directory root of the MXE Win64 cross environment.  Necessary
                         For Win64 cross-compiling only.

MATLAB_ROOT_WIN64 - Necessary for Matlab on Win64 cross build.  Location of the matlab WIN64 version to link against.

[Cross-building to OSX]

OSXCROSS_ROOT - Local directory root of the OSXCROSS OSX 64-bit cross environment.
                         Necessary for OSX cross-compiling only.

MATLAB_ROOT_MACI64 - Necessary for Matlab on OSX cross build.  Location of the matlab WIN64 version to link against.
