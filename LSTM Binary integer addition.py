
import numpy as np


class LSTM_Layer(object):
    
    def __init__(self, input_size, hidden_size):
        
        #LSTM size specification
        self.I_dim = input_size
        self.H_dim = hidden_size
        
        #Weight initialisation - Xavier initialisation
        self.W_xg = np.random.randn(self.H_dim,self.I_dim)/np.sqrt(self.H_dim+self.I_dim)
        self.W_hg = np.random.randn(self.H_dim,self.H_dim)/np.sqrt(self.H_dim+self.H_dim) 
        self.b_g = np.random.randn(self.H_dim)/np.sqrt(self.H_dim) 

        #i
        self.W_xi = np.random.randn(self.H_dim,self.I_dim)/np.sqrt(self.H_dim+self.I_dim) 
        self.W_hi = np.random.randn(self.H_dim,self.H_dim)/np.sqrt(self.H_dim+self.H_dim)
        self.b_i = np.random.randn(self.H_dim)/np.sqrt(self.H_dim) 

        #f
        self.W_xf = np.random.randn(self.H_dim,self.I_dim)/np.sqrt(self.H_dim+self.I_dim) 
        self.W_hf = np.random.randn(self.H_dim,self.H_dim)/np.sqrt(self.H_dim+self.H_dim) 
        self.b_f = np.random.randn(self.H_dim)/np.sqrt(self.H_dim) 

        #o
        self.W_xo = np.random.randn(self.H_dim,self.I_dim)/np.sqrt(self.H_dim+self.I_dim) 
        self.W_ho = np.random.randn(self.H_dim,self.H_dim)/np.sqrt(self.H_dim+self.H_dim) 
        self.b_o = np.random.randn(self.H_dim)/np.sqrt(self.H_dim) 
        
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def feedforward(self, X, seq_length):
        
        #Gate arguments, cell state and cell outputs
        self.g_input = np.zeros((seq_length,self.H_dim))
        self.i_input = np.zeros((seq_length,self.H_dim))
        self.f_input = np.zeros((seq_length,self.H_dim))
        self.o_input = np.zeros((seq_length,self.H_dim))

        self.H = np.zeros((seq_length,self.H_dim))
        self.C = np.zeros((seq_length,self.H_dim))

        self.Ht = np.zeros((seq_length,self.H_dim))
        self.Ct = np.zeros((seq_length,self.H_dim))
        
        #Forward pass

        for j in range(seq_length):
            
            x = X[:,j]

            self.Ht[j, :] = self.H[j-1, :] if j>0 else np.zeros((self.H_dim))

            self.g_input[j,:] = np.dot(self.W_xg,x)+np.dot(self.W_hg,self.Ht[j,:])+self.b_g
            self.i_input[j,:] = np.dot(self.W_xi,x)+np.dot(self.W_hi,self.Ht[j,:])+self.b_i
            self.f_input[j,:] = np.dot(self.W_xf,x)+np.dot(self.W_hf,self.Ht[j,:])+self.b_f
            self.o_input[j,:] = np.dot(self.W_xo,x)+np.dot(self.W_ho,self.Ht[j,:])+self.b_o

            self.g_t = np.tanh(self.g_input[j])
            self.i_t = self.sigmoid(self.i_input[j])
            self.f_t = self.sigmoid(self.f_input[j])
            self.o_t = self.sigmoid(self.o_input[j])

            self.Ct[j, :] = self.C[j-1, :] if j>0 else np.zeros((self.H_dim))

            self.C[j, :] = self.g_t*self.i_t+self.Ct[j,:]*self.f_t

            self.H[j, :] = np.tanh(self.C[j,:])*self.o_t

        #Gate outputs through time
        self.g = np.tanh(self.g_input)
        self.i = self.sigmoid(self.i_input)
        self.f = self.sigmoid(self.f_input)
        self.o = self.sigmoid(self.o_input)
        
        return self.H
    
    def backprop(self, X, deltas, seq_length):
        
        #Arrays to hold deltas
        self.d_H = np.zeros_like(self.H)
        self.d_C = np.zeros_like(self.C)

        self.d_I = np.zeros_like(self.i)
        self.d_G = np.zeros_like(self.g)
        self.d_F = np.zeros_like(self.f)
        self.d_O = np.zeros_like(self.o)

        self.d_Ht = np.zeros_like(self.H)
        self.d_Ct = np.zeros_like(self.C)

        self.d_I_input = np.zeros_like(self.i)
        self.d_G_input = np.zeros_like(self.g)
        self.d_F_input = np.zeros_like(self.f)
        self.d_O_input = np.zeros_like(self.o)

        self.d_W_xi = np.zeros_like(self.W_xi)
        self.d_W_hi = np.zeros_like(self.W_hi)
        self.d_b_i = np.zeros_like(self.b_i)

        self.d_W_xg = np.zeros_like(self.W_xg)
        self.d_W_hg = np.zeros_like(self.W_hg)
        self.d_b_g = np.zeros_like(self.b_g)

        self.d_W_xf = np.zeros_like(self.W_xf)
        self.d_W_hf = np.zeros_like(self.W_hf)
        self.d_b_f = np.zeros_like(self.b_f)

        self.d_W_xo = np.zeros_like(self.W_xo)
        self.d_W_ho = np.zeros_like(self.W_ho)
        self.d_b_o = np.zeros_like(self.b_o)
        
        for j in reversed(range(seq_length)):
        
            if j<seq_length-1:
                self.d_H[j,:] = deltas[j,:] + np.dot(self.W_hi,self.d_I_input[j+1,:]) + np.dot(self.W_ho,self.d_O_input[j+1,:]) 
                + np.dot(self.W_hf,self.d_F_input[j+1,:]) + np.dot(self.W_hg,self.d_G_input[j+1,:])
            else:
                self.d_H[j,:] = deltas[j,:]
            
            if j<seq_length-1:
                self.d_C[j,:] = self.d_H[j,:]*self.o[j,:]*(1-np.tanh(self.C[j,:])**2)+self.f[j+1,:]*self.d_C[j+1,:]
            else:
                self.d_C[j,:] = self.d_H[j,:]*self.o[j,:]*(1-np.tanh(self.C[j,:])**2)
            
            self.d_O[j,:] = self.d_H[j,:]*np.tanh(self.C[j,:])
            self.d_I[j,:] = self.d_C[j,:]*self.g[j,:]
            self.d_G[j,:] = self.d_C[j,:]*self.i[j,:]
            self.d_F[j,:] = self.d_C[j,:]*self.Ct[j,:]

            self.d_I_input[j,:] = self.d_I[j,:]*self.i[j,:]*(1-self.i[j,:])
            self.d_F_input[j,:] = self.d_F[j,:]*self.f[j,:]*(1-self.f[j,:])
            self.d_O_input[j,:] = self.d_O[j,:]*self.o[j,:]*(1-self.o[j,:])

            self.d_G_input[j,:] = self.d_G[j,:]*(1-self.g[j,:]**2)

            self.d_W_xi += np.outer(self.d_I_input[j,:],X[:,j])
            self.d_W_hi += np.outer(self.d_I_input[j,:],self.Ht[j,:])
            self.d_b_i += self.d_I_input[j,:]

            self.d_W_xg += np.outer(self.d_G_input[j,:],X[:,j])
            self.d_W_hg += np.outer(self.d_G_input[j,:],self.Ht[j,:])
            self.d_b_g += self.d_G_input[j,:]

            self.d_W_xf += np.outer(self.d_F_input[j,:],X[:,j])
            self.d_W_hf += np.outer(self.d_F_input[j,:],self.Ht[j,:])
            self.d_b_f += self.d_F_input[j,:]

            self.d_W_xo += np.outer(self.d_O_input[j,:],X[:,j])
            self.d_W_ho += np.outer(self.d_O_input[j,:],self.Ht[j,:])
            self.d_b_o += self.d_O_input[j,:]

        d_X = np.dot(self.W_xi.T,self.d_I_input.T)+np.dot(self.W_xf.T,self.d_F_input.T)
        +np.dot(self.W_xo.T,self.d_O_input.T)+np.dot(self.W_xg.T,self.d_G_input.T)
        
        w_upd_dict={}
        
        w_upd_dict['W_xi']=self.d_W_xi
        w_upd_dict['W_hi']=self.d_W_hi
        w_upd_dict['b_i']=self.d_b_i
        
        w_upd_dict['W_xg']=self.d_W_xg
        w_upd_dict['W_hg']=self.d_W_hg
        w_upd_dict['b_g']=self.d_b_g
        
        w_upd_dict['W_xf']=self.d_W_xf
        w_upd_dict['W_hf']=self.d_W_hf
        w_upd_dict['b_f']=self.d_b_f
        
        w_upd_dict['W_xo']=self.d_W_xo
        w_upd_dict['W_ho']=self.d_W_ho
        w_upd_dict['b_o']=self.d_b_o
        
        return  d_X, w_upd_dict
        
    def weight_update(self, w_upd_dict, alpha):
        
        self.W_xi += alpha*w_upd_dict['W_xi']
        self.W_hi += alpha*w_upd_dict['W_hi']
        self.b_i += alpha*w_upd_dict['b_i']

        self.W_xg += alpha*w_upd_dict['W_xg']
        self.W_hg += alpha*w_upd_dict['W_hg']
        self.b_g += alpha*w_upd_dict['b_g']

        self.W_xf += alpha*w_upd_dict['W_xf']
        self.W_hf += alpha*w_upd_dict['W_hf']
        self.b_f += alpha*w_upd_dict['b_f']

        self.W_xo += alpha*w_upd_dict['W_xo']
        self.W_ho += alpha*w_upd_dict['W_ho']
        self.b_o += alpha*w_upd_dict['b_o']


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


H_dim = 16
O_dim = 1
t=8 
alpha=0.5
eps=2000

W_hy = np.random.randn(O_dim,H_dim)/np.sqrt(O_dim+H_dim)
b_y = np.random.randn(O_dim)/np.sqrt(O_dim)

net = LSTM_Layer(2,H_dim)

for k in range(eps):
        a = np.random.randint(2**(t-1)-1)
        b = np.random.randint(2**(t-1)-1)
        c = a + b
        #Loop through input length
        X = input_prep(a,b)

        n = len(X.T)

        c_bin = int2bin(c)
        c_binary = np.zeros(n)
        c_binary[n-len(c_bin):]=c_bin
        c_binary[:n-len(c_bin)]=np.zeros(n-len(c_bin))
        
        #LSTM part
        H = net.feedforward(X,len(X.T))
        
        z = np.dot(W_hy,H.T)+b_y

        #Output 
        y = sigmoid(z)
        
        dy = c_binary[::-1] - y
        
        dz = dy*sigmoid_d(z)
        delta_h = np.outer(dz, W_hy)
        
        W_hy += alpha*np.dot(dz, H)

        b_y += alpha*np.sum(dz)
    
        _, updates = net.backprop(X, delta_h,len(X.T))
        net.weight_update(updates, alpha)


#Test LSTM on binary addition of numbers far greater than 2**8. Recall that training is done on
#adding integers in range 0-127. Below test demonstrates that the network truly has learned to 
#perform addition, by adding integers in range 2**15.

def test(num_cases, t):
    
    a = np.random.randint(2**(t-1)-1)
    b = np.random.randint(2**(t-1)-1)
    c = a + b

    X = input_prep(a,b)
    H = net.feedforward(X,len(X.T))

    z = np.dot(W_hy,H.T)+b_y

    #Output 
    y = sigmoid(z)

    print ("a: ", a, "b: ", b, "a+b:", c, "Network output: ", bin2int(np.round(y[0][::-1]).astype(int)))
        
    
test(1,32)


