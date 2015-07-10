#!/usr/bin/python3
import numpy as np
import scipy as sp
from numpy.linalg import norm
from sklearn.preprocessing import normalize

sigmaSq = 6

def genInitValues(numInp, numHidd, numOutp):
    inp = sp.rand(numInp)
    states = sp.rand(numHidd)
    outp = sp.rand(numOutp)
    N = numInp+numHidd+numOutp
    M = numHidd+numOutp
    density = 1/sp.sqrt(N)
    W = sp.sparse.rand(N, M, density)
    negValDensity = 0.3
    nM = sp.sign(sp.rand(N, M)-1+negValDensity)
    W = W.multiply(nM)
    W = normalize(W, axis=1, copy=False)
    return (inp, states, outp, sp.sparse.csr_matrix(W))

def step(inp, states, outp, W):
    S = np.concatenate([inp, states, outp])
    Xcurr = generateSparseActivationMatrix(S, W)
    actNext = sp.exp(-(1/2*sigmaSq)*norm(Xcurr-W,axis=1)**2)
    print(actNext.shape)
    (statesNext, outpNext) =np.split(actNext,states.shape)
    return (np.squeeze(normalize(statesNext)), np.squeeze(normalize(outpNext)))



def generateSparseActivationMatrix(S, W):
    return W.multiply(np.tile(S,[W.shape[1],1]).T)




if __name__ == '__main__':
    main()
