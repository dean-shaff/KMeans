import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

class KMeans(object):
    

    def __init__(self, **kwargs):
        """
        Initialize K means object. 
        args:
            X: the set of vectors that we want to compute clusters (None) 
            ndim: The number of dimensions for random initialization (None) 
            size: The size of the dataset for random initializaton (None)  
        """
        X = kwargs.get('X', None) 
        ndim = kwargs.get('ndim',None) 
        size = kwargs.get('size',None)
        if X != None:
            self.X = X
            self.ndim = self.X.shape[1]
            self.size = self.X.shape[0]
        else:
            assert ndim != None, "If X is None, then you need to specify a number of dimensions for random initialization"
            assert size != None, "If X is None, then you need to specify the size of dataset for random initialization"
            self.X = self.randomInitialize(ndim, size)
             
            self.ndim = ndim 
            self.size = size
        #print(self.X.shape) 
    def setX(self,Xnew):
        """
        Reset X 
        """
        self.X = Xnew
   
    def EuclideanDistance(self,x,y):
        """
        Calculate Euclidean distance 
        """ 
        return np.sqrt(np.sum((x-y)**2,axis=1))
    
    def centroid(self, X):
        """
        X is a vector whose rows corresponds to vectors in dataset  
        """
        return np.mean(X,axis=0)
    
    def randomInitialize(self,ndim,size):
        """
        initialize (size) random vectors, each of size (ndim) 
        """
        randos_mu = np.random.randn(ndim)
        randos_sigma = np.random.randn(ndim) 
        X = np.concatenate((randos_sigma[0]*np.random.randn(size,ndim)+randos_mu[0],
                            randos_sigma[1]*np.random.randn(size,ndim)+randos_mu[1]))
        return X 
    
    def calcClusters(self,nk,**kwargs):
        """
        args:
            nk: number of clusters to form 
        kwargs:
            nepoch: number of epochs to run the sucker for. (10)
            distFunc: The function to use to calculate distance (self.EuclideanDistance)  
        """
        #below i initialize the k vectors
        X = self.X  
        nepoch = kwargs.get('nepoch',10)
        distFunc = kwargs.get('distFunc',self.EuclideanDistance)  
        ks = np.zeros((nk, self.ndim))
        for i in xrange(self.ndim):
            min_coli = np.amin(X[:,i])
            max_coli = np.amax(X[:,i])
            #print(min_coli, max_coli)     
            ks[:,i] = np.random.uniform(min_coli,max_coli,nk)
        for epoch in xrange(nepoch):
            dist = np.ones((X.shape[0], nk)) 
            for i in xrange(nk):
                dist[:,i] = distFunc(X,ks[i])
            smallest_arg = np.argmin(dist,axis=1)
            #print(dist) 
            #print(smallest_arg) 
            #raw_input(">>> ")  
            for i in xrange(nk):
                cond = smallest_arg == i
                if X[cond].size == 0:
                    ks[i] = np.zeros(self.ndim)
                else:  
                    ks[i] = self.centroid(X[cond]) 
        return ks

    def crossValidate(self,n,nk,**kwargs):
        """
        cross validate K means clustering. 
        args:
            n: number of times to cross validate
            nk: number of clusters
        kwargs: to be passed to self.calcClusters  
        """
        ks = np.zeros((n,nk,self.ndim))
        for i in xrange(n):
            ks_i = self.calcClusters(nk, **kwargs)
            ks[i,:,:] = ks_i 
        ksMean = np.mean(ks,axis=0) 
        return ksMean, ks

if __name__ == '__main__':
    trainer = KMeans(ndim=2,size=10000)
    ksMean, ks= trainer.crossValidate(1,2,nepoch=5)
    #print(ks) 
    # print(k_init)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(trainer.X[:,0], trainer.X[:,1],c='r')
    ax.scatter(ksMean[:,0],ksMean[:,1],c='b')
    #ax.grid(True)
    plt.show()
