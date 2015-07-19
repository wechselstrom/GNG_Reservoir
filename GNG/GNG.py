#!/usr/bin/python3
import numpy as np
import scipy as sp
import scipy.spatial.distance



NaN = float('NaN')

class GNG:
    
    def __init__(self):
        self.NumOfEpochs = 600;
        self.NumOfSamples = 400;
        self.age_inc = 1;
        self.max_age = 200;
        self.max_nodes = 160;
        self.eb = .05;
        self.en = .005;
        self.Nlambda = 350;
        self.alpha = .5; 
        self.d = .99;
        self.RMSE = np.zeros(self.NumOfEpochs);
        self.distance_function = 'euclidean'
        self.nodes = None

    def fit(self,X):
        np.random.shuffle(X) ##randomize input sequence
        Ncounter = 0
        #step 0: initialization
        #start with two inputs as initial nodes
        self.nodes = X[-2:,:] ## initial nodes
        #make a initial Edge between the nodes
        self.edges = np.array([[0,1],[1,0]])
        #initialize age matrix
        self.ages = np.array([[NaN, 0], [0,NaN]])
        #initialize error vector
        self.error = np.array([0,0])
 
        for kk in range(self.NumOfEpochs):
            # Choose the next Input Training Vectors.
            nextblock = np.random.choice(X.shape[0],self.NumOfSamples)
            In = X[nextblock,:]
            for n in range(self.NumOfSamples):
                Ncounter += 1
                #step2 find nearest neighbors:
                inp = In[n,:]
                ds = np.squeeze(scipy.spatial.distance.cdist( inp.reshape((1,inp.shape[0])), self.nodes, self.distance_function))
                winner, secondbest, *_ = np.argsort(ds)
                #Steps 3-6. Increment the age of all edges emanating from the winner node
                self.edgeManagement(inp, ds, (winner,secondbest))
                #Step 7. Dead Node Removal Procedure.
                self.removeUnconnected()
                # Step 8. Node Insertion Procedure.
                if (Ncounter % self.Nlambda == 0 and self.nodes.shape[0] < self.max_nodes):
                    self.addNewNeuron()
                # Step 9. Finally, decrease the error of all units.
                self.error = self.d*self.error

    def edgeManagement(self, inp, ds, winners):
        # Step 3. Increment the age of all edges emanating from s1. 
        fst, snd = winners
        fstneighbors = self.neighborsOf(fst)
        self.ages[fstneighbors,fst]+=self.age_inc
        self.ages[fst,fstneighbors] = self.ages[fstneighbors,fst]
        #step4. Add the squared distance to a local error counter variable:
        self.error[fst] += ds[fst]**2
        # Step 5. Move s1 and its topological neighbors towards î.
        self.nodes[fst,:] += self.eb*(inp-self.nodes[fst,:])
        self.nodes[fstneighbors,:] += self.en*(inp-self.nodes[fstneighbors,:])
        # Step 6.
        # If s1 and s2 are connected by an edge, set the age of this edge to zero.
        # If such an edge does not exist, create it.
        self.edges[fst,snd] = 1
        self.edges[fst,snd] = 1
        self.ages[fst,snd] = 0
        self.ages[fst,snd] = 0
        #step 7. remove eges older than max_age
        mask = self.ages > self.max_age
        self.edges[mask] = 0
        self.ages[mask] = NaN
    def removeUnconnected(self):
        indices = np.squeeze(np.argwhere(np.sum(self.edges,axis=1))) #find columns with one non zero entry
        mask = np.ix_(indices,indices)
        self.edges = self.edges[mask]
        self.ages = self.ages[mask]
        self.nodes = self.nodes[indices,:]
        self.error = self.error[indices]

        pass

    def addNewNeuron(self):
        q =np.argmax(self.error) # neuron with maximal error
        mEneighbors = self.neighborsOf(q)
        p = mEneighbors[np.argmax(self.error[mEneighbors])] #neighbor with maximal error of neuron with maximal error
        self.nodes = np.vstack([self.nodes, self.nodes[(q,p),:].mean(axis=0)]) # insert new neuron between both
        self.edges[q,p] = 0
        self.edges[p,q] = 0
        self.ages[q,p] = NaN
        self.ages[p,q] = NaN
        #increase size in edge and age matrix
        (i,j) = self.edges.shape
        en = np.zeros((i+1,j+1))
        en[:i,:j] = self.edges
        an = np.zeros((i+1,j+1))
        an[:i,:j]  = self.ages
        
        #add initial edges
        en[p,i] = 1
        en[q,i] = 1
        en[i,p] = 1
        en[i,q] = 1
        self.edges = en
        #add initial ages

        an[q,i] = 0
        an[i,q] = 0
        an[p,i] = 0
        an[i,p] = 0
        self.ages = an
    
        self.error[q] *= self.alpha
        self.error[p] *= self.alpha

        self.error = np.append(self.error,self.error[q])

    def neighborsOf(self,q):
        return np.squeeze(np.argwhere(self.edges[:,q]), axis=1)
