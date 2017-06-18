
# coding: utf-8

# In[30]:

import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

#Define network variables
alpha = 0.0001
beta = 0.0001
H = 4
gamma = 1

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

while True:

    #Actor
    a_w_l = np.random.randn(H)

    #Critic
    c_w_l = np.random.randn(H)

    #Initialise state
    state = env.reset()

    #Training initialisation
    running_reward = 0
    reward_sum = 0
    num_eps=0
    t=0
    training_progress = []
    done = True
    
    for i in range(1000):

        while True:
            if done:
                t=0
                num_eps+=1
                running_reward = 0.99*running_reward + 0.01*reward_sum
                training_progress.append(running_reward)
                reward_sum=0
                state = env.reset()

                #Sample an action a
                a_l = np.dot(a_w_l, state)
                logp = sigmoid(a_l)

                action = 1 if 0.5<logp else 0

                #Q(s,a)
                q = np.dot(c_w_l, state)

            #Observe the reward and transition state s' (observation)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            t+=1

            #Sample an action a' 
            a_l = np.dot(a_w_l, next_state)
            next_logp = sigmoid(a_l)

            next_action = 1 if 0.5<next_logp else 0

            #Q(s',a')
            next_q = np.dot(c_w_l, next_state)


            #Parameter updates
            td_error = reward + gamma*next_q - q

            pg = (1-logp)*q

            #Backpropagate
            #Actor
            a_d_w_l = np.dot(state.T,pg)
            a_w_l += alpha*a_d_w_l

            #Critic
            c_d_w_l = np.dot(state.T,td_error)

            c_w_l -= beta*c_d_w_l

            state = next_state
            action = next_action
            q=next_q
            logp = next_logp

            if done:
                
                break
                   
    if running_reward>195:
        print (a_w_l)
        break           

plt.plot(training_progress)
plt.ylabel('100 episode average reward')
plt.xlabel('Episodes')
plt.show()

