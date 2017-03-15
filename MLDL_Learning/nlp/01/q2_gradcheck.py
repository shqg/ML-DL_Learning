#! /usr/bin/python
# coding=utf-8
'''
Created on Feb 28, 2017
@author: gengsq2
'''
import numpy as np
import random

def gradcheck_naive(f, x):
    
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)
    
    h = 1e-4
    
    # iterate over all indexs in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        ix=it.multi_index
        
        
        # replace with return to avoid side effects
        x[ix] += h # increment by h
        random.setstate(rndstate)  
        fxh, _ = f(x) # evalute f(x + h)
        x[ix] -= 2 * h # restore to previous value (very important!)
        random.setstate(rndstate)  
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h
    
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext() # Step to next dimension
    print "check passed"
def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""

if __name__ == "__main__":
    sanity_check()
