import cPickle
import gzip
import numpy as np
import random


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

test_datax = np.array([b[0] for b in test_data]).reshape(10000,784)
test_datay = np.array([b[1] for b in test_data]).reshape(10000,1)


#Network parameters
eta = 3
beta = 0
lmbda = 0
batch_size=20
epochs = 30

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def network(training_data,eta,beta,lmbda,batch_size,test_data=None):
    
    w_h = np.random.randn(30,784)
    w_l = np.random.randn(10,30)
    b_h = np.random.randn(30,1)
    b_l = np.random.randn(10,1)
    
    for j in xrange(epochs):
        
        np.random.shuffle(training_data)
        
        train = np.array([b[0] for b in training_data]).reshape(50000,784).T
        target = np.array([b[1] for b in training_data]).reshape(50000,10).T
        
        for k in xrange(0,50000,batch_size):
            
            batch = np.array(training_data[k:k+batch_size])
            
            batch_train = train[:,k:k+batch_size]
            batch_target = target[:,k:k+batch_size]
            
            #Compute hidden layer activations
            z_h = np.dot(w_h,batch_train) + b_h
            a_h = sigmoid(z_h)

            #Compute output layer activations
            z_l = np.dot(w_l,a_h) + b_l
            a_l = sigmoid(z_l)

            #Compute delta l
            d_l = (a_l-batch_target)*sigmoid_deriv(z_l)
            
            # Compute w_l update
            d_w_l = np.dot(d_l,a_h.T)
        
            # Compute b_l update
            d_b_l = d_l.sum(axis=1).reshape(10,1)
            
            # Compute delta h
            d_h = np.dot(w_l.T,d_l)*sigmoid_deriv(z_h)

            #Compute w_h update
            d_w_h = np.dot(d_h,batch_train.T)
            
            # Compute b_h update
            d_b_h = d_h.sum(axis=1).reshape(30,1)
            
            #Update w_l, b_l, w_h, b_h
            w_l = w_l*(1-lmbda*eta/50000) - eta*d_w_l/batch_size
            b_l = b_l - eta*d_b_l/batch_size
            w_h = w_h*(1-lmbda*eta/50000) - eta*d_w_h/batch_size
            b_h = b_h - eta*d_b_h/batch_size
            
        pred = [(np.argmax(sigmoid(np.dot(w_l,sigmoid(np.dot(w_h,x.reshape(784,1))+b_h))+b_l)),y) for (x,y) in zip(test_datax,test_datay)]
        print sum(int(x == y) for (x, y) in pred)

network(training_data,eta,beta,lmbda,batch_size,test_data)
