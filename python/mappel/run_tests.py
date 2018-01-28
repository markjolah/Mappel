# 
# Mark J. Olah (mjo\@cs.unm.edu)
# 2018
# Mappel python tests

import numpy
#import mappel
from _Gauss1DMLE import Gauss1DMLE 

def run_tests():
    M = Gauss1DMLE(8,1);
    n=10;

    theta = M.sample_prior(n);
    print("Theta: ", theta)
    ims = M.simulate_image(theta);

    llh = M.objective_llh(ims,theta);
    print(llh)

#if __name__=="__main__":
    #run_tests()
