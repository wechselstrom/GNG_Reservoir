#!/usr/bin/python3.4
import numpy as np
import scipy as sp
from numpy.linalg import norm
from sklearn.preprocessing import normalize
import sklearn.linear_model

sigmaSq = 6


def genInitValues(numInp, numHidd):
    N = numInp+numHidd
    M = numHidd
    density = 1/sp.sqrt(N)
    W = sp.sparse.rand(N, M, density)
    negValDensity = 0.3
    nM = sp.sign(sp.rand(N, M)-1+negValDensity)
    W = W.multiply(nM)
    W = normalize(W, axis=0, copy=False).T
    return (sp.sparse.csr_matrix(W))


def oneOfKtoStr(ook):
    if sp.sparse.issparse(ook):
        ook = ook.toarray()
    indices = np.argmax(ook,axis=1)
    s = ''
    for i in indices:
        if i==0:
            s+='.'
        elif i==1:
            s+=','
        elif i==2:
            s+=' '
        elif i==3:
            s+='\''
        else: s+=chr(i+ord('a')-4)
    return(s)
def oneOfK(string):
    i = 0
    row = []
    col = []
    data = []
    d = 0
    for x in string:
        if x.islower():
            d = ord(x)-ord('a')+4
        elif x.isupper():
            d = ord(x)-ord('A')+4
        elif x=='.':
            d = 0
        elif x == ',':
            d = 1
        elif x == ' ':
            d = 2
        elif x == '\'':
            d = 3
        else:
            continue
        row.append(i)
        col.append(d)
        data.append(1.0)
        i+=1
    return sp.sparse.csr_matrix((data, (row,col)), shape=(i,30))

def step(inp, states, W):
    S = np.concatenate([np.squeeze(inp), states])
    Xcurr = generateSparseActivationMatrix(S, W)
    Diff = (Xcurr-W)
    Diff.data = Diff.data**2
    Xnorm = np.array(Diff.sum(axis=1))
    statesNext = np.squeeze(sp.exp(-(1/2*sigmaSq)*Xnorm))
    return statesNext

def computeOutputs(states,RR):
    return sp.squeeze(RR.predict(states))

def runWithStimuli(inp, W, RR=None, states=None):
    if states is None:
        states = sp.zeros(W.shape[0])

    for i in range(inp.shape[0]):
        states = step(inp[i,:].toarray().T, states, W)
        if RR is None:
            yield states
        else:
            out = computeOutputs(states, RR)
            yield out,states

def trainWeights(inp, outp, W, a):
    states = runStimulitoArray(inp,W)
    RR = sklearn.linear_model.Ridge(alpha=a)
    if sp.sparse.issparse(outp):
        outp=outp.toarray()
    RR.fit(states, outp)
    return RR

def runStimulitoArray(stim, W, RR=None):
    if RR is None:
        return sp.array([val for val in runWithStimuli(stim, W)]) 
    else:
        x = [val for val in runWithStimuli(stim, W, RR)]
        out, states = zip(*x)
        return sp.array(out),sp.array(states)



def generateSparseActivationMatrix(S, W):
    Wc = W.copy()
    Wc.data = S[Wc.indices]
    Wc = normalize(Wc,axis=1, norm='l2')
    return Wc

def testcase2(filename, trainsamples, numStates, lastChars, alpha):
    with open(filename,'r') as f:
        cont = f.read()
        inp1 = oneOfK(cont[:trainsamples])
        *x1, y1 = genlistOfShiftedStrings(inp1)
        x1 = sp.sparse.hstack(x1).tocsr()

        W = genInitValues(x1.shape[1], numStates)
        RR = trainWeights(x1, y1, W, alpha)
        states = sp.zeros(W.shape[0])
        x = oneOfK('lala').toarray().reshape(1,120)
        s = ''
        for i in range(1000):
            states = step(x,states,W)
            out = computeOutputs(states,RR)
            x[0,0:90] = x[0,30:120]
            x[0,90:120] = out
            s += oneOfKtoStr([out])
        return s

genlistOfShiftedStrings = lambda inp: [inp[i:-(lastChars-i)] for i in range(0,lastChars)]

def testcase1(filename, trainsamples, testsamples, numStates, lastChars, alpha):
    with open(filename,'r') as f:
        cont = f.read()

        inp1 = oneOfK(cont[:trainsamples])
        *x1, y1 = genlistOfShiftedStrings(inp1)
        x1 = sp.sparse.hstack(x1).tocsr()

        inp2 = oneOfK(cont[trainsamples:trainsamples+testsamples])
        *x2, y2 = genlistOfShiftedStrings(inp2)
        x2 = sp.sparse.hstack(x2).tocsr()
        print('finished generation of testsamples')
        W = genInitValues(x1.shape[1], numStates)
        print('initialized network randomly')
        RR = trainWeights(x1, y1, W, alpha)
        print('finished training')
        out1, _ = runStimulitoArray(x1, W, RR)
        out2, _ = runStimulitoArray(x2, W, RR)
        print('ran on testset')
        return oneOfKtoStr(y1),oneOfKtoStr(out1), oneOfKtoStr(y2), oneOfKtoStr(out2)



if __name__ == '__main__':
    filename = 'shakespeare.txt'
    trainsamples = 15000
    lastChars = 5
    numStates = 3000
    testsamples = 2000
    alpha=1e-4
    y1,out1,y2,out2 = testcase1(filename, trainsamples, testsamples, numStates, lastChars, alpha)
    print('\noriginal traindata:\n%s\nreconstructed traindata:\n%s\noriginal testdata:\n%s\nrecosntructed testdata:\n%s\n' % (y1,out1,y2,out2))

    s = testcase2(filename, trainsamples, numStates, lastChars, alpha)
    print(s)
