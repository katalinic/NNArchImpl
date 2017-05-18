
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt



#Other functions

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_d(z):
    return sigmoid(z)*(1-sigmoid(z))


#Converts integer to binary representation, stored as an array
def int2bin(m):
    return np.asarray([int(x) for x in bin(m)[2:]])

#Converts a binary representation in an array to an integer
def bin2int(n):
    s = 0
    for i in range(len(n)):
        s += n[i]*2**(len(n)-i-1)
    return s

#Prepare LSTM input by stacking the binary representations of the two numbers being added
def input_prep(a,b):
    a_bin = int2bin(a)
    b_bin = int2bin(b)
    
    L = len(b_bin) if len(b_bin)>=len(a_bin) else len(a_bin)
    
    #Compare their lengths; pad longer by 1 0, pad shorter to same length as longer
    a_out = np.zeros(L+1)
    b_out = np.zeros(L+1)
    
    a_out[1+L-len(a_bin):] = a_bin
    a_out[:1+L-len(a_bin)] = np.zeros(1+L-len(a_bin))
    
    b_out[1+L-len(b_bin):] = b_bin
    b_out[:1+L-len(b_bin)] = np.zeros(1+L-len(b_bin))
    
    return np.vstack((a_out[::-1], b_out[::-1]))


def LSTM(eps, alpha, inp, hidden, output):
    #Model variables

    alpha = alpha

    I_dim = inp
    H_dim = hidden
    O_dim = output

    #Track performance
    correct = [0]

    #Model weights and biases
    
    #g
    W_xg = np.random.randn(H_dim,I_dim)/np.sqrt(H_dim+I_dim)
    W_hg = np.random.randn(H_dim,H_dim)/np.sqrt(H_dim+H_dim) 
    b_g = np.random.randn(H_dim)/np.sqrt(H_dim) 

    #i
    W_xi = np.random.randn(H_dim,I_dim)/np.sqrt(H_dim+I_dim) 
    W_hi = np.random.randn(H_dim,H_dim)/np.sqrt(H_dim+H_dim)
    b_i = np.random.randn(H_dim)/np.sqrt(H_dim) 

    #f
    W_xf = np.random.randn(H_dim,I_dim)/np.sqrt(H_dim+I_dim) 
    W_hf = np.random.randn(H_dim,H_dim)/np.sqrt(H_dim+H_dim) 
    b_f = np.random.randn(H_dim)/np.sqrt(H_dim) 

    #o
    W_xo = np.random.randn(H_dim,I_dim)/np.sqrt(H_dim+I_dim) 
    W_ho = np.random.randn(H_dim,H_dim)/np.sqrt(H_dim+H_dim) 
    b_o = np.random.randn(H_dim)/np.sqrt(H_dim) 

    #y
    W_hy = np.random.randn(O_dim,H_dim)/np.sqrt(O_dim+H_dim)
    b_y = np.random.randn(O_dim)/np.sqrt(O_dim)
      
    t=8

    for k in range(eps):
        a = np.random.randint(2**(t-1)-1)
        b = np.random.randint(2**(t-1)-1)
        c = a + b

        #Loop through input length
        X = input_prep(a,b).T

        n = len(X)

        c_bin = int2bin(c)
        c_binary = np.zeros(n)
        c_binary[n-len(c_bin):]=c_bin
        c_binary[:n-len(c_bin)]=np.zeros(n-len(c_bin))

        #Arrays to store inputs to gates, cell state and hidden outputs through time

        g_input = np.zeros((n,H_dim))
        i_input = np.zeros((n,H_dim))
        f_input = np.zeros((n,H_dim))
        o_input = np.zeros((n,H_dim))

        H = np.zeros((n,H_dim))
        C = np.zeros((n,H_dim))

        Ht = np.zeros((n,H_dim))
        Ct = np.zeros((n,H_dim))

        #Forward pass

        for j in range(n):

            x = X[j]

            Ht[j, :] = H[j-1, :] if j>0 else np.zeros((H_dim))

            g_input[j,:] = np.dot(W_xg,x)+np.dot(W_hg,Ht[j,:])+b_g
            i_input[j,:] = np.dot(W_xi,x)+np.dot(W_hi,Ht[j,:])+b_i
            f_input[j,:] = np.dot(W_xf,x)+np.dot(W_hf,Ht[j,:])+b_f
            o_input[j,:] = np.dot(W_xo,x)+np.dot(W_ho,Ht[j,:])+b_o

            g_t = np.tanh(g_input[j])
            i_t = sigmoid(i_input[j])
            f_t = sigmoid(f_input[j])
            o_t = sigmoid(o_input[j])

            Ct[j, :] = C[j-1, :] if j>0 else np.zeros((H_dim))

            C[j, :] = g_t*i_t+Ct[j,:]*f_t

            H[j, :] = np.tanh(C[j,:])*o_t

        #Gate outputs through time
        g = np.tanh(g_input)
        i = sigmoid(i_input)
        f = sigmoid(f_input)
        o = sigmoid(o_input)

        #Output layer activations
        z = np.dot(W_hy,H.T)+b_y

        #Output 
        y = sigmoid(z)

        #Arrays to hold deltas
        d_H = np.zeros_like(H)
        d_C = np.zeros_like(C)

        d_W_hy = np.zeros_like(W_hy)
        d_b_y = np.zeros_like(b_y)
        
        d_I = np.zeros_like(i)
        d_G = np.zeros_like(g)
        d_F = np.zeros_like(f)
        d_O = np.zeros_like(o)

        d_Ht = np.zeros_like(H)
        d_Ct = np.zeros_like(C)

        d_I_input = np.zeros_like(i)
        d_G_input = np.zeros_like(g)
        d_F_input = np.zeros_like(f)
        d_O_input = np.zeros_like(o)

        d_W_xi = np.zeros_like(W_xi)
        d_W_hi = np.zeros_like(W_hi)
        d_b_i = np.zeros_like(b_i)

        d_W_xg = np.zeros_like(W_xg)
        d_W_hg = np.zeros_like(W_hg)
        d_b_g = np.zeros_like(b_g)

        d_W_xf = np.zeros_like(W_xf)
        d_W_hf = np.zeros_like(W_hf)
        d_b_f = np.zeros_like(b_f)

        d_W_xo = np.zeros_like(W_xo)
        d_W_ho = np.zeros_like(W_ho)
        d_b_o = np.zeros_like(b_o)

        #Backward pass
        dy = c_binary[::-1] - y

        for j in reversed(range(n)):
        
            dz = dy[:,j]*sigmoid_d(z[:,j])
                    
            d_W_hy += np.outer(dz, H[j,:])
            d_b_y += dz
     
            if j<n-1:
                d_H[j,:] = np.outer(dy[:,j]*sigmoid_d(z[:,j]), W_hy) + np.dot(W_hi,d_I_input[j+1,:])
                + np.dot(W_ho,d_O_input[j+1,:]) + np.dot(W_hf,d_F_input[j+1,:]) + np.dot(W_hg,d_G_input[j+1,:])
            else:
                d_H[j,:] = np.outer(dy[:,j]*sigmoid_d(z[:,j]), W_hy)
            
            if j<n-1:
                d_C[j,:] = d_H[j,:]*o[j,:]*(1-np.tanh(C[j,:])**2)+f[j+1,:]*d_C[j+1,:]
            else:
                d_C[j,:] = d_H[j,:]*o[j,:]*(1-np.tanh(C[j,:])**2)
            
            
            d_O[j,:] = d_H[j,:]*np.tanh(C[j,:])
            d_I[j,:] = d_C[j,:]*g[j,:]
            d_G[j,:] = d_C[j,:]*i[j,:]
            d_F[j,:] = d_C[j,:]*Ct[j,:]

            d_I_input[j,:] = d_I[j,:]*i[j,:]*(1-i[j,:])
            d_F_input[j,:] = d_F[j,:]*f[j,:]*(1-f[j,:])
            d_O_input[j,:] = d_O[j,:]*o[j,:]*(1-o[j,:])

            d_G_input[j,:] = d_G[j,:]*(1-g[j,:]**2)

            d_W_xi += np.outer(d_I_input[j,:],X[j,:])
            d_W_hi += np.outer(d_I_input[j,:],Ht[j,:])
            d_b_i += d_I_input[j,:]

            d_W_xg += np.outer(d_G_input[j,:],X[j,:])
            d_W_hg += np.outer(d_G_input[j,:],Ht[j,:])
            d_b_g += d_G_input[j,:]

            d_W_xf += np.outer(d_F_input[j,:],X[j,:])
            d_W_hf += np.outer(d_F_input[j,:],Ht[j,:])
            d_b_f += d_F_input[j,:]

            d_W_xo += np.outer(d_O_input[j,:],X[j,:])
            d_W_ho += np.outer(d_O_input[j,:],Ht[j,:])
            d_b_o += d_O_input[j,:]

        W_hy += alpha*d_W_hy
        b_y += alpha*d_b_y
                          
        W_xi += alpha*d_W_xi
        W_hi += alpha*d_W_hi
        b_i += alpha*d_b_i

        W_xg += alpha*d_W_xg
        W_hg += alpha*d_W_hg
        b_g += alpha*d_b_g

        W_xf += alpha*d_W_xf
        W_hf += alpha*d_W_hf
        b_f += alpha*d_b_f

        W_xo += alpha*d_W_xo
        W_ho += alpha*d_W_ho
        b_o += alpha*d_b_o

        r = 1 if c==bin2int(np.round(y[0][::-1]).astype(int)) else 0

        correct.append(correct[k]+r)

    percent_correct = [(correct[k*100+100]-correct[k*100])/100 for k in range(round(eps/100)-1)]

    plt.plot(correct)
    plt.show()

    plt.plot(percent_correct)
    plt.show()
    
LSTM(10000,0.5,2,32,1)

