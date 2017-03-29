#Plots circular towers defined as arrays [x, y, r] passed to the circles array
#Used in the "Universal approximation theorem visualisation with a single hidden layer" blog post
#Author: Goran Katalinic

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

circles = np.array([[0.7,0.1,0.1],[0.2,0.4,0.1],[0.5,0.8,0.1]])
# circles = np.array([[0.5,0.5,0.1]])
x=y=np.arange(0,1,0.01)
def par_gen(x0,y0,r,theta):
    th = theta + 0.000001 if np.mod(theta,np.pi)==0 else theta
    return -np.cos(th), -np.sin(th), y0*np.sin(th)+x0*np.cos(th)+r

def circle_NN(circles):
    z = np.zeros((x.size,y.size))
    b1=w2=np.array([])
    w1=np.array([]).reshape(0,2)
    for i in xrange(0,60):
        theta = i * np.pi/30
        w1 = np.vstack((w1,100*np.array([par_gen(x0,y0,r,theta)[:-1] for x0,y0,r in circles]),                        100*np.array([par_gen(x0,y0,r,theta)[:-1] for x0,y0,r in circles])))
        b1 = np.concatenate((b1,100*np.array([par_gen(x0,y0,r,theta)[-1] for x0,y0,r in circles]),             100*(np.array([par_gen(x0,y0,r,theta)[-1] for x0,y0,r in circles])-2*circles[:,-1])))
        w2 = np.concatenate((w2,np.ones(len(circles)),-np.ones(len(circles))))
    b2 = np.array([-50])
    z = nn_2input(x, y, w1, w2, b1, b2)    

    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, z)
    ax.set_zlim3d(0,np.amax(z))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
circle_NN(circles)
