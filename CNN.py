import pickle
import gzip
import numpy as np
import random
import time

def load_data():
    f = gzip.open('/mnist.pkl.gz','rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data

def vectorised_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def softmax_overflow_batch(z):
    m = np.max(z,axis=1).reshape(-1,1)
    sf = np.exp(z-m)
    nm = np.sum(sf,axis=1).reshape(-1,1)
    return sf/nm

def mpool_indices(input_side, v):
    all_inds = np.asarray([[[j,i] for i in range(input_side)] for j in range(input_side)])
    t = int(np.floor((input_side)/v))
    ind = np.arange(t)*v
    corner_inds = np.asarray([[[j,i] for i in ind] for j in ind]).reshape(-1,2)
    m_inds = np.asarray([[all_inds[i:i+v,j:j+v] for j in ind] for i in ind]).reshape(-1,2,2,2)
    return corner_inds, m_inds

def convolution_indices(input_side, l, s):
    all_inds = np.asarray([[[j,i] for i in range(input_side)] for j in range(input_side)])
    t = int(np.floor((input_side-l)/s))+1
    ind = np.arange(t)*s
    alt_conv_inds = np.row_stack([np.row_stack([np.row_stack(all_inds[i:i+l,j:j+l]) for j in ind]) for i in ind])
    return alt_conv_inds

def convolution_batch(M, batch, conv_indices, l):
    return M[:,conv_indices[:,0],conv_indices[:,1]].reshape(batch,-1,l,l)

maps = 20
l = 5 #window side length
s = 1 #stride length
v = 2 #max-pooling window
n_F = 100 #fully-connected layer
n_O = 10 #output
f = 10000 #show progress every f examples
batch_size = 10
input_side = 28 #MNIST images are 28x28
r = int(np.floor((input_side-l)/s))+1 #side of each feature map
m = int(r/v) #side of max pool
alpha = 0.005

train_d, _, _ = load_data()
train = train_d[0].reshape(-1,input_side,input_side)
y = train_d[1]
target = np.array([vectorised_result(y) for y in y]).reshape(-1,10)

#Xavier weight initialisation
Wy = np.random.randn(n_O,n_F)/np.sqrt(n_F+n_O)
by = np.random.randn(n_O)/np.sqrt(n_O)

Wf = np.random.randn(maps,n_F,m*m)/np.sqrt(n_F+m*m)
bf = np.random.randn(n_F)/np.sqrt(n_F)

W = np.random.randn(maps,l,l)/np.sqrt(l+l)
b = np.random.randn(maps)

correct = 0

#Constant inputs for convolution and max pooling steps
c_inds = convolution_indices(input_side, l, s)
corner_inds, mpool_inds = mpool_indices(r,v)

y = range(batch_size)
x = range(maps)

#Used in backpropagation
diag_bat = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
diag_bat2 = np.fliplr(diag_bat)
diag_bat3 = np.column_stack((diag_bat2,diag_bat2))

epochs = 40

for epoch in range(epochs):
    
    start = time.time()
    
    for k in range(0,len(train),batch_size):

        train_batch = train[k:k+batch_size]
        target_batch = target[k:k+batch_size]

        #Convolution
        C = convolution_batch(train_batch, batch_size, c_inds, l)

        #Feature map
        F = (np.einsum('ijk,mljk->mil',W,C)+b[np.newaxis,:,np.newaxis]).reshape(batch_size,maps,r,r)

        #ReLu
        F[F<0]=0
        
        #Pre-Max pooling step
        F_pools = F[:,:,mpool_inds[:,:,:,0],mpool_inds[:,:,:,1]]

        #Indices of max value in each max pool
        max_idx = F_pools.reshape(F_pools.shape[0],F_pools.shape[1],F_pools.shape[2],-1).argmax(3)

        all_inds = np.unravel_index(max_idx, F_pools[0,0,0,:,:].shape)

        pooled_inds = np.column_stack((all_inds[0].ravel(),all_inds[1].ravel())).reshape(batch_size,maps,-1,2)

        final_inds = corner_inds+pooled_inds
        
        #Max pooling
        M = F[:,:,final_inds[:,:,:,0],final_inds[:,:,:,1]][diag_bat3[:,0],diag_bat3[:,1],diag_bat3[:,2],diag_bat3[:,3]].reshape(batch_size,maps,-1)

        #Fully connected layer
        Z = np.einsum('ijk,lik->lj',Wf,M)+bf

        #ReLu
        Z_a = Z.copy()
        Z_a[Z<0]=0

        #Output layer
        Y = np.dot(Z_a,Wy.T)+by

        Y_a = softmax_overflow_batch(Y)

        #Track training progress with correct classifications
        correct += int(np.sum(np.argmax(Y_a,axis=1)==np.argmax(target_batch,axis=1)))
        
        #Backpropagation
        
        #Output error
        dY_a = -(Y_a-target_batch)

        dY = dY_a

        #Fully connected layer to output layer updates
        d_Wy = np.dot(dY.T,Z_a)
        d_by = np.sum(dY,axis=0)
        
        #Fully connected layer errors
        dZ = np.dot(dY,Wy)
        
        #Backprop ReLu
        dZ[Z<=0]=0

        #Max pooling to fully connected layer updates
        d_Wf = np.einsum('ij,ikl->kjl',dZ, M)
        d_bf = np.sum(dZ,axis=0)

        #Max pooling errors
        dM = np.einsum('ijk,lj->lik',Wf,dZ)
        
        #Backprop feature map errors
        dF = np.zeros(F.size)

        #To avoid explicit loops and advanced indexing when backpropagating max-pool deltas to feature maps
        flat_inds = final_inds.reshape(-1,2)

        flat_inds_xy_form = [flat_inds[:,0],flat_inds[:,1]] 

        td = np.ravel_multi_index(flat_inds_xy_form,F[0,0].shape)

        joint_inds = np.column_stack((np.repeat(range(batch_size*maps), m*m),td))

        tens_inds = [joint_inds[:,0],joint_inds[:,1]]

        dF_full_flat = np.ravel_multi_index(tens_inds,dF.reshape(batch_size*maps,-1).shape)
        
        #Backprop max-deltas to feature map
        dF[dF_full_flat]=dM.ravel()

        dF = dF.reshape(F.shape)

        dF[F<=0]=0

        #Flatten dF for backprop to conv layer
        dF_flat = dF.reshape(batch_size,dF.shape[1],-1)

        #Backprop dW
        dW = np.einsum('hli,hijk->ljk',dF_flat,C)
        db = np.sum(np.sum(dF_flat,axis=0),axis=1)

        #Update weights and biases
        Wf += alpha*d_Wf/batch_size
        bf += alpha*d_bf/batch_size
        Wy += alpha*d_Wy/batch_size
        by += alpha*d_by/batch_size

        W += alpha*dW/batch_size
        b += alpha*db/batch_size

        if k%f==0 and k>0: 
            training_accuracy = correct/f
            if training_accuracy<=1: 
                print(time.time()-start, training_accuracy)
            correct=0
        

