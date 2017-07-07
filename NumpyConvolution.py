class Convolution(object):
    
    def __init__(self, v, l, s, f):
        
        self.l = l
        self.v = v
        self.f = f
        
        t = int(np.floor((v-l)/s))+1
        u = np.arange(t)*s
        
        self.row = np.tile(np.repeat(u,t),l*l)+np.repeat(np.arange(l),t*t*l)
        self.col = np.tile(u,t*l*l)+np.tile(np.repeat(np.arange(l),t*t),l)
        
        self.x = np.tile(self.row,f)
        self.y = np.tile(self.col,f)
        self.z = np.tile(np.repeat(np.arange(l*l), t*t), f)
        
    def fwd_pass(self, M, W, b):
        
        self.C = M[self.row,self.col].reshape(self.l*self.l,-1)
        
        return np.dot(W,self.C)+b[:,np.newaxis]
    
    def back_pass(self, dz, W, b, back_to_input=False):

        dW = np.dot(dz,self.C.T)
        
        db = np.sum(dz,axis=1)
        
        if back_to_input==True:
            
            dM = np.zeros((self.f,self.l*self.l,self.v,self.v))

            dzM = W[...,np.newaxis]*dz[:,np.newaxis,:]

            dM[:,self.z, self.x, self.y] = dzM.ravel()
        
            return np.sum(dM,axis=1), dW, db
        
        else: return dW, db
