import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
import gzip

# Loads MNIST data 
def load_data():
    f = gzip.open('YOUR LOCATION/mnist.pkl.gz','rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorised_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorised_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

training_data , test_data, d2= load_data()



# 8x8 image extraction
from itertools import chain

def extract_88(x,s,l,d):
    row = [x[s+i*d:s+i*d+l] for i in xrange(l)]
    return np.asarray([i for i in chain.from_iterable(row)])

# Stack arrays to create training dataset, shape (64,50000)
new_training_data = np.asarray([extract_88(x,np.random.randint(0,561),8,28) for x in training_data[0]]).T

# Only retain 8x8 patches which are not all black (i.e. no feature present)
zero = (np.max(new_training_data, axis=0) != 0.0)

td_z = new_training_data[:,zero]


# Sparse Autoencoder implementation
def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

eta = 0.07
beta = 15
lmbda = 0.001
batch_size=10
epochs = 500
rho = 0.01

w_f = np.sqrt(6./129.)

def network(data,eta,beta,lmbda,batch_size):
    N = 64
    H = 25
    w_h = w_f*(2*np.random.rand(H,N)-1)
    w_l = w_f*(2*np.random.rand(N,H)-1)
    b_h = np.zeros((H,1))
    b_l = np.zeros((N,1))
    
    batch_data = np.array(data)
    
    for j in xrange(epochs):
        
        np.random.shuffle(batch_data.T)
        
        for k in xrange(0,len(data[0])-np.mod(len(data[0]),batch_size),batch_size):
            
            
            batch = batch_data[:,k:k+batch_size]
           
            #Compute hidden layer activations
            z_h = np.dot(w_h,batch) + b_h
            a_h = sigmoid(z_h)

            rho_h = np.mean(a_h,1).reshape((H,1))

            #Compute output layer activations
            z_l = np.dot(w_l,a_h) + b_l
            a_l = sigmoid(z_l)

            C = np.sum([np.dot((a_l[:,i]-batch[:,i]).T,a_l[:,i]-batch[:,i]) for i in xrange(batch_size)])/(2*batch_size)

            #Compute delta l
            d_l = (a_l-batch)*sigmoid_deriv(z_l)

            # Compute w_l update
            d_w_l = np.dot(d_l,a_h.T)

            # Compute b_l update
            d_b_l = d_l.sum(axis=1).reshape(N,1)

            # Compute delta h
            d_h = (np.dot(w_l.T,d_l)+beta*(-rho/rho_h + (1.-rho)/(1.-rho_h)))*sigmoid_deriv(z_h)
            
            #Compute w_h update
            d_w_h = np.dot(d_h,batch.T)
            
            # Compute b_h update
            d_b_h = d_h.sum(axis=1).reshape(H,1)
            
            #Update w_l, b_l, w_h, b_h
            w_l = w_l*(1-lmbda*eta/len(data[0])) - eta*d_w_l/batch_size
            b_l = b_l - eta*d_b_l/batch_size
            w_h = w_h*(1-lmbda*eta/len(data[0])) - eta*d_w_h/batch_size
            b_h = b_h - eta*d_b_h/batch_size
        
    return w_l, b_l, w_h, b_h
     
network_params = network(td_z[:,:10000],eta,beta,lmbda,batch_size)

# Visualise learned features
w_n = network_params[2]

x_s = w_n/np.linalg.norm(w_n,axis=1).reshape(25,1)

fig, ax = plt.subplots(5,5,figsize=(5,5))

for i,a in enumerate(ax.flatten()):
    
    # Convert to 8x8
    w88 = np.reshape(x_s[i],(8,8))
    a.imshow(w88, cmap=cm.Greys) 

plt.show()
