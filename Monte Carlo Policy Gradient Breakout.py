import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Initialise parameters

#Learning rate
alpha = 0.0001

#Discount rate
gamma = 0.99

#Decay rate for rmsprop
decay = 0.99

#Batch-size
batch = 10

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#Pre-process input (210,160,3) to 80x80
def pre_process(M):
    M = M[40:200]
    M = M[::2,::2,0]
    M[M==142]=0
    M[M!=0]=1
    return M.astype(np.float).ravel()

#Discounted reward
def discounted_reward(r):
    d_r = np.zeros_like(r).astype(np.float)
    d_r[-1]=r[-1]
    for i in range(len(r)-1):
        d_r[-i-2]=r[-i-2]+gamma*d_r[-i-1]
    return d_r

#Initialise environment
env = gym.make('Breakout-v0')

#Network elements
w_h = np.random.randn(200,6400)/np.sqrt(6400)
w_l = np.random.randn(200)/np.sqrt(200)


#Train the agent

num_eps = 0

running_reward = None
reward_sum = 0

#Initialise training
observation = env.reset()
x_prev = None

#input layer, hidden layer, reward, gradient at each time step
t_x, t_h, t_r, t_g = [], [], [], []

#cumulative gradient for batch rmsprop
s_w_l, s_w_h = np.zeros_like(w_l), np.zeros_like(w_h)
rms_w_l, rms_w_h = np.zeros_like(w_l), np.zeros_like(w_h)

t = 0 

while True: 
#     env.render()
    
    #Current state; obs will be updated with each action
    x_cur = pre_process(observation)
    #Network input is difference between obs at t and t-1
    x = x_cur - x_prev if x_prev is not None else np.zeros(6400)
    x_prev=x_cur

    #Using input x, calculate probability of taking an action
    #4 = right, 5 = left
    h = np.dot(w_h, x)
    h[h<0]=0
    l = np.dot(w_l, h)
    logp = sigmoid(l)

    #Choose an action based on this probability
    action = 4 if np.random.uniform()<logp else 5

    if action == 4: running_right+=1
    if action == 5: running_left+=1

    g = 1 - logp if action==4 else 0 - logp

    #Take the action
    observation, reward, done, info = env.step(action)
    
    reward_sum += reward

    #info outputs 

    #Store values 
    t_x.append(x)
    t_h.append(h)
    t_r.append(reward)
    t_g.append(g)
    
    if done: #episode finished
        num_eps+=1
        
        #Stack episode values together
        ep_x = np.vstack(t_x) 
        ep_h = np.vstack(t_h)
        ep_r = np.vstack(t_r)
        ep_g = np.vstack(t_g)
        
        t_x, t_h, t_r, t_g = [], [], [], []
        
        #Discount rewards for the episode and normalise
        discounted_epr = discounted_reward(ep_r)

        if np.std(discounted_epr) != 0:
            discounted_epr = (discounted_epr-np.mean(discounted_epr))/np.std(discounted_epr)
        
        # Policy gradient
        ep_logp = ep_g * discounted_epr 
         
        #Backpropagate the gradient
        d_w_l = np.dot(ep_h.T,ep_logp).ravel()

        d_h = np.outer(ep_logp, w_l)
        d_h[ep_h<=0]=0
        d_w_h = np.dot(d_h.T,ep_x)
        
        s_w_l += d_w_l
        s_w_h += d_w_h
        
        
        #Perform rmsprop every batch-number of episodes
        if np.mod(num_eps,batch)==0:
            
            rms_w_l = decay * rms_w_l + (1-decay) * s_w_l**2
            rms_w_h = decay * rms_w_h + (1-decay) * s_w_h**2
            w_l += alpha * s_w_l / (np.sqrt(rms_w_l)+1e-8)
            w_h += alpha * s_w_h / (np.sqrt(rms_w_h)+1e-8)
            
            s_w_l, s_w_h = np.zeros_like(w_l), np.zeros_like(w_h)

        #Training process done
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        reward_sum = 0
        
        observation = env.reset() # reset env
        x_prev = None
