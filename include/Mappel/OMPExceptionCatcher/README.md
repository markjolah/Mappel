# OMPExceptionCatcher
A lightweight class for managing C++ exception handling strategies in OpenMP code.

## Motivation
OpenMP code must catch any exceptions that may have been thrown before exiting the OpenMP block.
This class acts as lightweight wrapper that allows an arbitrary function or lambda expression to be run
safely and efficiently in OMP even if it might throw exceptions.  We employ one of 4 possible strategies
as determined By the OMPExceptionCatcher::Strategies enum.
 
 ## Excepton Catching Strategy's
 * `OMPExceptionCatcher::Strategies::DoNotTry` -- Don't even try,  this is a null op to completely disable
                                               this class's effect.
 * `OMPExceptionCatcher::Strategies::Continue` -- Catch exceptions and keep going
 * `OMPExceptionCatcher::Strategies::Abort`    -- Catch exceptions and abort
 * `OMPExceptionCatcher::Strategies::RethrowFirst`  -- Re-throws first exception thrown by any thread
 
 
 
 ## Including OMPExceptionCatcher in your OpenMP project
Since OMPExceptionCatcher is header-only, he easiest way to use it is via the [git subrepo](https://github.com/ingydotnet/git-subrepo) plugin.  Unlike the traditional `git submodule` command, `git subrepo` is transparent to other users of your repository, and solves many of the irksome issues prevalent with the submodule approach.  Follow the [git subrepo install guide](https://github.com/ingydotnet/git-subrepo#installation-instructions) to install on your development machine.

Then to add OMPExceptionCatcher,
```
> cd $MY_REPOS
> git subrepo pull https://github.com/markjolah/OMPExceptionCatcher include/where/ever/OMPExceptionCactcher
```
 
 ## Example useage:
 ~~~.cxx
 #include <OMPExceptionCatcher/OMPExceptionCatcher.h>
 OMPExceptionCatcher catcher(OMPExceptionCatcher<>::Strategies::Continue);
 #pragma omp parallel for
 for(int n=0; n < N; n++) 
     catcher.run([&]{ my_ouput(n)=do_my calulations(args(n)); }
 catcher.rethrow(); //Required only if you ever might use RethrowFirst strategy
 ~~~
 
 
 # License
 * Author: Mark J. Olah
 * Email: (mjo@cs.unm DOT edu)
 * Copyright: 2019
 * LICENSE: Apache 2.0.  See [LICENSE](https://github.com/markjolah/OMPExceptionCatcher/blob/master/LICENSE) file.
